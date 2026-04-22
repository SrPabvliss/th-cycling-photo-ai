"""
Modal training script — PP-OCRv5 mobile Phase 1: Synthetic pretraining.

Uses PaddlePaddle + PaddleOCR recognition training pipeline.

Usage:
    modal run --detach scripts/modal_train_ppocr_phase1.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase1_ppocr ./experiments/
"""

import modal

app = modal.App("ocr-phase1-ppocr")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "paddlepaddle-gpu>=3.0.0",
        "paddleocr>=2.9.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.9.0",
    )
    .run_commands(
        "pip install 'git+https://github.com/PaddlePaddle/PaddleOCR.git@main'",
    )
)

VOLUME_PATH = "/data"
SEED = 42


@app.function(
    image=image,
    gpu="a10g",
    timeout=14400,
    volumes={VOLUME_PATH: volume},
)
def train_ppocr_phase1():
    import io
    import json
    import os
    import random
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    from PIL import Image

    random.seed(SEED)
    np.random.seed(SEED)

    print("PP-OCRv5 Phase 1 — Synthetic pretraining")

    synth_lmdb = Path(VOLUME_PATH) / "ocr" / "synthetic" / "lmdb"
    if not synth_lmdb.exists():
        print("ERROR: Synthetic data not found.")
        return

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_ppocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Convert LMDB to PaddleOCR format ----
    # PaddleOCR needs: image_dir + label_file (path\ttext)
    ppocr_dir = Path(VOLUME_PATH) / "ocr" / "synthetic" / "ppocr_format"

    if not (ppocr_dir / "train_label.txt").exists():
        print("Converting LMDB to PaddleOCR format...")
        imgs_dir = ppocr_dir / "images"
        imgs_dir.mkdir(parents=True, exist_ok=True)

        env = lmdb_lib.open(str(synth_lmdb), readonly=True, lock=False)
        with env.begin() as txn:
            n = int(txn.get("num-samples".encode()).decode())

        labels = []
        # Split: 95% train, 5% val
        n_train = int(n * 0.95)

        with env.begin() as txn:
            for idx in range(n):
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

                fname = f"syn_{idx:07d}.jpg"
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                # Resize to PP-OCR default: 48x192
                img = img.resize((192, 48), Image.LANCZOS)
                img.save(imgs_dir / fname, quality=95)

                labels.append((fname, label))

                if (idx + 1) % 50000 == 0:
                    print(f"  Converted {idx+1}/{n}")

        # Write label files
        train_labels = labels[:n_train]
        val_labels = labels[n_train:]

        with open(ppocr_dir / "train_label.txt", "w") as f:
            for fname, label in train_labels:
                f.write(f"images/{fname}\t{label}\n")

        with open(ppocr_dir / "val_label.txt", "w") as f:
            for fname, label in val_labels:
                f.write(f"images/{fname}\t{label}\n")

        env.close()
        print(f"  Train: {len(train_labels)}, Val: {len(val_labels)}")

        from modal import Volume
        volume.commit()
    else:
        print("PaddleOCR format already cached")

    # ---- 2. Create digit-only dictionary ----
    dict_path = ppocr_dir / "digits_dict.txt"
    if not dict_path.exists():
        with open(dict_path, "w") as f:
            for d in "0123456789":
                f.write(f"{d}\n")

    # ---- 3. Create PaddleOCR config ----
    config_content = f"""
Global:
  debug: false
  use_gpu: true
  epoch_num: 50
  log_smooth_window: 20
  print_batch_step: 100
  save_model_dir: {str(output_dir)}
  save_epoch_step: 10
  eval_batch_step: [0, 1000]
  cal_metric_during_train: true
  checkpoints: null
  pretrained_model: null
  save_inference_dir: null
  use_visualdl: false
  character_dict_path: {str(dict_path)}
  max_text_length: 4
  infer_mode: false
  use_space_char: false
  distributed: false
  save_res_path: {str(output_dir / 'predicts.txt')}

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform: null
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 4

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss: null
    - SARLoss: null

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {str(ppocr_dir)}
    label_file_list:
      - {str(ppocr_dir / 'train_label.txt')}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug: null
      - MultiLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 192]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_sar
            - length
            - valid_ratio
    loader:
      shuffle: true
      batch_size_per_card: 128
      drop_last: true
      num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {str(ppocr_dir)}
    label_file_list:
      - {str(ppocr_dir / 'val_label.txt')}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 192]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_sar
            - length
            - valid_ratio
    loader:
      shuffle: false
      drop_last: false
      batch_size_per_card: 128
      num_workers: 2
"""
    config_path = output_dir / "config.yml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # ---- 4. Train ----
    print("\nStarting PP-OCR training...")
    start_time = time.time()

    import subprocess
    result = subprocess.run(
        [
            "python", "-m", "tools.train",
            "-c", str(config_path),
        ],
        cwd="/usr/local/lib/python3.11/site-packages/paddleocr",
        capture_output=False,
        timeout=14000,
    )

    elapsed = (time.time() - start_time) / 60

    # If PaddleOCR tools.train doesn't work directly, try alternative
    if result.returncode != 0:
        print(f"  PaddleOCR tools.train failed (exit {result.returncode})")
        print("  Trying alternative: paddleocr CLI...")

        # Alternative: use PaddleOCR Python API
        try:
            import paddle
            from ppocr.modeling.architectures import build_model
            from ppocr.utils.save_load import load_model

            print(f"  PaddlePaddle version: {paddle.__version__}")
            print(f"  GPU available: {paddle.device.is_compiled_with_cuda()}")

            # Save what we have
            summary = {
                "phase": "synthetic",
                "model": "ppocr_mobile",
                "status": "training_api_issue",
                "total_time_min": elapsed,
                "error": "PaddleOCR training pipeline needs investigation",
            }
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            print(f"  PaddlePaddle import error: {e}")
            summary = {
                "phase": "synthetic",
                "model": "ppocr_mobile",
                "status": "failed",
                "error": str(e),
            }
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
    else:
        print(f"\nPP-OCR training complete! Time: {elapsed:.1f} min")
        summary = {
            "phase": "synthetic",
            "model": "ppocr_mobile",
            "status": "completed",
            "total_time_min": elapsed,
            "seed": SEED,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    volume.commit()
    print(f"Results saved to volume")


@app.local_entrypoint()
def main():
    train_ppocr_phase1.remote()
