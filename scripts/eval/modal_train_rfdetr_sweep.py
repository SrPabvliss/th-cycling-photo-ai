"""Barrido RF-DETR (ESWA): una corrida por invocación o tanda en paralelo. Espejo de modal_train_yolo_sweep.py.

Receta congelada = la de producción (ADR-016 Run 1, scripts/modal_train_adr016_run1_896.py), ver
D1-auditoria-receta-rfdetr.md: epochs 80, batch 16, grad_accum 1, lr 2e-4, lr_encoder 2e-5, weight_decay 1e-4,
multi_scale, do_random_resize_via_padding, EMA, early stopping paciencia 12 / min_delta 0.003, resolución nativa
de cada variante (N 384, S 512, M 576, L 704). Versión única rfdetr==1.10.1 (D-2). GPU A100 40 GB (decisión §8).

Datos: sube una vez data/v4_c1_coco (build_coco_dataset.py):
    uv run modal volume put cycling-photo-ai-sweep-vol data/v4_c1_coco datasets/v4_c1_coco

Entrenar:
    uv run modal run --detach scripts/eval/modal_train_rfdetr_sweep.py --models "small,medium,large" --seeds "42"
    (--resolution N para forzar una resolución distinta de la nativa; múltiplo de 56)

Salida por corrida (runs/rfdetr_<variant>_<res>_s<seed>/): checkpoints de rfdetr, run_meta.json (versiones, GPU,
tiempo, hashes, mAP de rfdetr en val), y dt_val.json / dt_test.json: detecciones COCO en píxeles originales a
umbral 0.001 para eval_boxes.py --dt-json (evaluador único con YOLO).
"""

from __future__ import annotations

from pathlib import Path

import modal

RFDETR_VERSION = "1.10.1"
GPU_TRAIN = "A100-40GB"
VOL_NAME = "cycling-photo-ai-sweep-vol"
VOLUME_PATH = "/vol"
NATIVE = {"nano": 384, "small": 512, "medium": 576, "large": 704}

app = modal.App("eswa-rfdetr-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"rfdetr[train]=={RFDETR_VERSION}", "pycocotools>=2.0.8", "pyyaml>=6.0.2", "pandas>=2.2.0", "supervision>=0.27.0")
    .add_local_file("experiments/split_v4/split.csv", "/root/split.csv")
)

RECIPE = dict(epochs=80, batch_size=16, grad_accum_steps=1, lr=2e-4, lr_encoder=2e-5, weight_decay=1e-4,
              multi_scale=True, do_random_resize_via_padding=True, use_ema=True,
              early_stopping=True, early_stopping_patience=12, early_stopping_min_delta=0.003)


@app.function(image=image, gpu=GPU_TRAIN, timeout=4 * 3600, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768)
def train(variant: str, resolution: int | None, seed: int, epochs: int | None = None, fixed: bool = False,
          noflip: bool = False) -> dict:
    import hashlib
    import json
    import time

    import torch
    import rfdetr
    from importlib.metadata import version as _v
    from PIL import Image

    ds = Path(VOLUME_PATH) / "datasets" / "v4_c1_coco"
    assert (ds / "train" / "_annotations.coco.json").exists(), "upload data/v4_c1_coco first"
    res = resolution or NATIVE[variant]
    run_name = f"rfdetr_{variant}_{res}_s{seed}" + ("_fx" if fixed else "") + ("_noflip" if noflip else "")
    out_dir = Path(VOLUME_PATH) / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[variant]
    gpu_name = torch.cuda.get_device_name(0)
    print(f"rfdetr {_v('rfdetr')} torch {torch.__version__} GPU {gpu_name} run {run_name} res {res}")

    torch.manual_seed(seed)
    model = cls(num_classes=1, resolution=res)
    recipe = dict(RECIPE)
    if epochs:
        recipe["epochs"] = epochs
    if fixed:
        # Protocolo común de la final (2026-09-23): épocas fijas, sin parada temprana; checkpoint_best_ema = mejor época en val
        recipe["early_stopping"] = False
    if noflip:
        # Ablación de volteo (P-87, 2026-09-25): AUG_CONFIG por defecto de rfdetr es solo HorizontalFlip p 0.5; {} lo quita
        recipe["aug_config"] = {}
    t0 = time.time()
    model.train(dataset_dir=str(ds), output_dir=str(out_dir), seed=seed, **recipe)
    train_s = time.time() - t0

    # Mejor checkpoint EMA → detecciones COCO en val y test a umbral 0.001 (píxeles originales)
    best = out_dir / "checkpoint_best_ema.pth"
    if not best.exists():
        best = out_dir / "checkpoint_best_total.pth"
    m = cls(num_classes=1, resolution=res, pretrain_weights=str(best))
    m.optimize_for_inference() if hasattr(m, "optimize_for_inference") else None
    for split, folder in (("val", "valid"), ("test", "test")):
        paths = sorted(p for p in (ds / folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        dets = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            d = m.predict(img, threshold=0.001)
            for (x1, y1, x2, y2), c, s in zip(d.xyxy.tolist(), d.class_id.tolist(), d.confidence.tolist()):
                dets.append({"image_id": i, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        json.dump(dets, open(out_dir / f"dt_{split}.json", "w"))
        print(f"{split}: {len(paths)} imgs, {len(dets)} dets")
    meta = {"run_name": run_name, "family": "rfdetr", "variant": variant, "resolution": res, "seed": seed,
            "rfdetr": _v("rfdetr"), "torch": torch.__version__, "gpu": gpu_name, "train_seconds": round(train_s, 1),
            "recipe": recipe, "split_sha256": hashlib.sha256(open("/root/split.csv", "rb").read()).hexdigest(),
            "params_M": round(sum(p.numel() for p in m.model.model.parameters()) / 1e6, 3) if hasattr(m, "model") and hasattr(m.model, "model") else None,
            "best_checkpoint": best.name}
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    volume.commit()
    print(json.dumps(meta, indent=2))
    return meta


@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume}, cpu=4, memory=16384)
def latency(run_names: list[str]) -> dict:
    """Protocolo v2 (igual que YOLO): un contenedor, orden intercalado, 3 repeticiones, mediana. GPU pura fp32 lote 1
    con tensor preparado a la resolución nativa (20 calentamientos, 100 medidas) y servicio por foto real (predict:
    decode + resize + inferencia, sin NMS) sobre 25 fotos ≥ 2000 px y 25 pequeñas de test."""
    import json
    import random
    import time

    import numpy as np
    import rfdetr
    import torch
    from PIL import Image

    def load(r):
        meta = json.load(open(Path(VOLUME_PATH) / "runs" / r / "run_meta.json"))
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[meta["variant"]]
        m = cls(num_classes=1, resolution=meta["resolution"], pretrain_weights=str(Path(VOLUME_PATH) / "runs" / r / meta["best_checkpoint"]))
        return m, meta["resolution"]

    def bench(m, res, half):
        net = m.model.model.cuda().eval()
        x = torch.rand(1, 3, res, res, device="cuda")
        ctx = torch.autocast("cuda", dtype=torch.float16) if half else torch.no_grad()
        with torch.no_grad(), ctx:
            for _ in range(20):
                net(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                t0 = time.perf_counter(); net(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean()), "std_ms": float(ts.std())}

    allp = sorted((Path(VOLUME_PATH) / "datasets" / "v4_c1_coco" / "test").glob("*.jpg"))
    big = [p for p in allp if max(Image.open(p).size) >= 2000][:25]
    small = [p for p in allp if max(Image.open(p).size) < 2000][:25]
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_big": len(big), "n_small": len(small), "reps": 3, "runs": {}}
    models = {r: load(r) for r in run_names}
    acc = {r: {"fp32": [], "fp16_autocast": [], "svc_big": [], "svc_small": []} for r in run_names}
    order = list(run_names)
    for rep in range(3):
        random.Random(rep).shuffle(order)
        for r in order:
            m, res = models[r]
            acc[r]["fp32"].append(bench(m, res, False))
            acc[r]["fp16_autocast"].append(bench(m, res, True))
            for p in big[:5] + small[:5]:
                m.predict(Image.open(p).convert("RGB"), threshold=0.4)
            for key, subset in (("svc_big", big), ("svc_small", small)):
                ts = []
                for p in subset:
                    t0 = time.perf_counter(); m.predict(Image.open(p).convert("RGB"), threshold=0.4); ts.append((time.perf_counter() - t0) * 1000)
                acc[r][key].append({"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))})
    for r in run_names:
        res = {k: {"p50_ms_median_of_reps": float(np.median([x["p50_ms"] for x in v])), "p50_ms_reps": [x["p50_ms"] for x in v],
                   "p95_ms_median_of_reps": float(np.median([x["p95_ms"] for x in v]))} for k, v in acc[r].items()}
        out["runs"][r] = res
        json.dump(res, open(Path(VOLUME_PATH) / "runs" / r / "latency_l4_v2.json", "w"), indent=2)
        print(r, json.dumps(res))
    volume.commit()
    return out


@app.local_entrypoint()
def main(models: str = "medium", seeds: str = "42", resolution: int = 0, epochs: int = 0, latency_runs: str = "", fixed: bool = False, noflip: bool = False):
    if latency_runs:
        latency.remote([r.strip() for r in latency_runs.split(",") if r.strip()]); return
    jobs = [(m.strip(), resolution or None, int(s), epochs or None, fixed, noflip) for m in models.split(",") for s in seeds.split(",")]
    print(f"lanzando {len(jobs)}: {[f'{j[0]}_{j[1] or NATIVE[j[0]]}_s{j[2]}' for j in jobs]}")
    for meta in train.starmap(jobs):
        print(f"done {meta['run_name']}: {meta['train_seconds']:.0f}s")
