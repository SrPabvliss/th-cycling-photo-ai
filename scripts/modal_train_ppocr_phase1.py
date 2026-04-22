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
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .run_commands(
        # PaddlePaddle GPU with CUDA 12 + cuDNN
        "pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html",
    )
    .pip_install(
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.9.0",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/PaddleOCR",
        "cd /opt/PaddleOCR && pip install -r requirements.txt",
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

    # ---- 4. Train using cloned PaddleOCR repo ----
    print("\nStarting PP-OCR training...")
    start_time = time.time()

    import sys
    import subprocess

    # Add PaddleOCR to path
    PPOCR_DIR = "/opt/PaddleOCR"
    sys.path.insert(0, PPOCR_DIR)

    result = subprocess.run(
        [
            sys.executable,
            f"{PPOCR_DIR}/tools/train.py",
            "-c", str(config_path),
        ],
        cwd=PPOCR_DIR,
        capture_output=True,
        text=True,
        timeout=14000,
    )

    elapsed = (time.time() - start_time) / 60

    # Print output
    if result.stdout:
        # Print last 50 lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-50:]:
            print(f"  {line}")

    if result.returncode != 0:
        print(f"\n  PP-OCR training failed (exit {result.returncode})")
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-20:]:
                print(f"  STDERR: {line}")

        summary = {
            "phase": "synthetic",
            "model": "ppocr_mobile",
            "status": "failed",
            "total_time_min": elapsed,
            "error": result.stderr[-500:] if result.stderr else "unknown",
        }
    else:
        print(f"\n  PP-OCR training complete! Time: {elapsed:.1f} min")

        # Find best model accuracy from logs
        best_acc = 0.0
        if result.stdout:
            for line in result.stdout.split("\n"):
                if "best_acc" in line.lower() or "acc:" in line.lower():
                    # Try to extract accuracy
                    import re
                    match = re.search(r'acc[:\s]+([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        try:
                            acc = float(match.group(1))
                            if acc > best_acc:
                                best_acc = acc
                        except ValueError:
                            pass

        summary = {
            "phase": "synthetic",
            "model": "ppocr_mobile",
            "status": "completed",
            "total_time_min": elapsed,
            "best_acc": best_acc,
            "seed": SEED,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"Results saved to volume")


@app.local_entrypoint()
def main():
    train_ppocr_phase1.remote()
