"""Barrido de detectores YOLO (ESWA): un entrenamiento por invocación, receta congelada.

Workspace Modal: eswa-eval-2 (adrianisrael-616). Volumen: cycling-photo-ai-sweep-vol.
Datos: las fotos se suben UNA vez (data/v4_c5/images, 1,1 GB); c2 y c1 solo suben labels y
data.yaml, y la función copia las imágenes dentro del volumen la primera vez:

    uv run modal volume put cycling-photo-ai-sweep-vol data/v4_c5 datasets/v4_c5
    uv run modal volume put cycling-photo-ai-sweep-vol data/v4_c2/labels datasets/v4_c2/labels
    uv run modal volume put cycling-photo-ai-sweep-vol data/v4_c2/data.yaml datasets/v4_c2/data.yaml
    (igual para v4_c1)

Entrenar (una configuración):

    MODAL_PROFILE=eswa-eval-2 uv run modal run --detach scripts/eval/modal_train_yolo_sweep.py \
        --model yolo11m.pt --classes c2 --imgsz 640 --seed 42

Descargar resultados:

    MODAL_PROFILE=eswa-eval-2 uv run modal volume get cycling-photo-ai-sweep-vol runs/<run_name> experiments/sweep_yolo/

El nombre de corrida es <model>_<classes>_<imgsz>_s<seed>. Guarda args.yaml, results.csv,
best.pt, la validación COCO por clase sobre val y sobre test (predictions.json + metrics.json)
y run_meta.json (versión de ultralytics, torch, GPU, tiempo de entrenamiento, hash de la receta y del split).
"""

from __future__ import annotations

from pathlib import Path

import modal

ULTRALYTICS_VERSION = "8.4.160"
import os
GPU_TRAIN = os.environ.get("YOLO_GPU", "A100-40GB")   # entrenamiento (decisión 2026-09-22 §8); latencia se mide en L4 con latency_gpu.py
# YOLO_GPU=A100-80GB para variantes l a 1024 con lote 32 (26l se queda sin memoria en 40 GB; ultralytics bajaría el lote a 16 y rompería el protocolo). Se declara, como en D-FINE.
VOL_NAME = "cycling-photo-ai-sweep-vol"
VOLUME_PATH = "/vol"

app = modal.App("eswa-yolo-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"ultralytics=={ULTRALYTICS_VERSION}", "pycocotools>=2.0.8", "pyyaml>=6.0.2", "pandas>=2.2.0")
    .add_local_file("configs/eval/recipe_v4.yaml", "/root/recipe_v4.yaml")
    .add_local_file("experiments/split_v4/split.csv", "/root/split.csv")
)


@app.function(image=image, gpu=GPU_TRAIN, timeout=4 * 3600, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768)
def train(model: str, classes: str, imgsz: int, seed: int, epochs: int | None = None, fixed: bool = False,
          fliplr: float | None = None) -> dict:
    import hashlib
    import json
    import time
    from pathlib import Path

    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO

    recipe = yaml.safe_load(open("/root/recipe_v4.yaml"))
    if epochs:
        recipe["epochs"] = epochs  # solo para pruebas de humo
    if fixed:
        # Protocolo común de la final (2026-09-23): épocas fijas, sin parada temprana; best.pt = mejor época en val
        recipe["patience"] = recipe["epochs"]
    if fliplr is not None:
        # Ablación de volteo (P-87, 2026-09-25): la receta congelada lleva fliplr 0.0; aquí se fuerza otro valor
        recipe["fliplr"] = float(fliplr)
    import shutil
    ds_root = Path(VOLUME_PATH) / "datasets"
    data_yaml = ds_root / f"v4_{classes}" / "data.yaml"
    assert data_yaml.exists(), f"missing {data_yaml}; upload labels + data.yaml first"
    if not (data_yaml.parent / "images").exists():
        src_imgs = ds_root / "v4_c5" / "images"
        assert src_imgs.exists(), "upload data/v4_c5 (images) first"
        print(f"copying images {src_imgs} -> {data_yaml.parent / 'images'}")
        shutil.copytree(src_imgs, data_yaml.parent / "images")
        volume.commit()
    # data.yaml lleva la ruta local del autor; reescribir al volumen
    d = yaml.safe_load(open(data_yaml))
    d["path"] = str(data_yaml.parent)
    data_vol = data_yaml.parent / "data_vol.yaml"
    yaml.safe_dump(d, open(data_vol, "w"), sort_keys=False)

    run_name = f"{Path(model).stem}_{classes}_{imgsz}_s{seed}" + ("_fx" if fixed else "") + (f"_flip{int(round(fliplr * 10))}" if fliplr is not None else "")
    out_root = Path(VOLUME_PATH) / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"ultralytics {ultralytics.__version__} torch {torch.__version__} GPU {gpu_name} run {run_name}")

    t0 = time.time()
    yolo = YOLO(model)
    yolo.train(data=str(data_vol), imgsz=imgsz, seed=seed, project=str(out_root), name=run_name,
               exist_ok=True, device=0, **recipe)
    train_s = time.time() - t0
    run_dir = out_root / run_name
    best = run_dir / "weights" / "best.pt"

    metrics = {}
    for split in ["val", "test"]:
        m = YOLO(str(best)).val(data=str(data_vol), split=split, imgsz=imgsz, batch=16, device=0,
                               save_json=True, plots=False, project=str(run_dir), name=f"eval_{split}", exist_ok=True)
        per_class = {}
        for i, cname in m.names.items():
            if i in m.box.ap_class_index:
                k = list(m.box.ap_class_index).index(i)
                per_class[cname] = {"AP50": float(m.box.ap50[k]), "AP50_95": float(m.box.ap[k]),
                                    "P": float(m.box.p[k]), "R": float(m.box.r[k])}
        metrics[split] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "mAP75": float(m.box.map75),
                          "per_class": per_class, "speed_ms": m.speed}
    meta = {
        "run_name": run_name, "model": model, "classes": classes, "imgsz": imgsz, "seed": seed,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "gpu": gpu_name,
        "train_seconds": round(train_s, 1), "fixed_epochs": fixed, "fliplr": recipe.get("fliplr"), "patience": recipe["patience"], "recipe_sha256": hashlib.sha256(open("/root/recipe_v4.yaml", "rb").read()).hexdigest(),
        "split_sha256": hashlib.sha256(open("/root/split.csv", "rb").read()).hexdigest(),
        "params_M": round(sum(p.numel() for p in YOLO(str(best)).model.parameters()) / 1e6, 3),
        "metrics": metrics,
    }
    json.dump(meta, open(run_dir / "run_meta.json", "w"), indent=2)
    volume.commit()
    print(json.dumps(meta, indent=2))
    return meta


@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume}, cpu=4, memory=16384)
def latency(run_names: list[str], imgsz: int = 640, n_photos: int = 100) -> dict:
    """R-6 en el hardware de despliegue: GPU pura fp32 y FP16 (lote 1, 20 calentamientos, 100 medidas, p50/p95)
    y tiempo de servicio por foto real (decode + letterbox + inferencia + NMS) sobre n_photos de test."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from ultralytics import YOLO

    def bench(model, half):
        x = torch.rand(1, 3, imgsz, imgsz, device="cuda")
        net = model.model.cuda().eval()
        net = net.half() if half else net.float()
        x = x.half() if half else x
        with torch.no_grad():
            for _ in range(20):
                net(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                t0 = time.perf_counter(); net(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean()), "std_ms": float(ts.std())}

    # Conjunto fijo: 25 fotos grandes (>= 2000 px) y 25 pequeñas del test, ordenadas por nombre; mismas para todos.
    from PIL import Image
    allp = sorted((Path(VOLUME_PATH) / "datasets" / "v4_c5" / "images" / "test").glob("*.jpg"))
    big = [p for p in allp if max(Image.open(p).size) >= 2000][:25]
    small = [p for p in allp if max(Image.open(p).size) < 2000][:25]
    photos = big + small
    import random
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "imgsz": imgsz, "n_big": len(big), "n_small": len(small), "reps": 3, "runs": {}}
    models = {r: YOLO(str(Path(VOLUME_PATH) / "runs" / r / "weights" / "best.pt")) for r in run_names}
    acc = {r: {"fp32": [], "fp16": [], "svc_big": [], "svc_small": []} for r in run_names}
    order = list(run_names)
    for rep in range(3):
        random.Random(rep).shuffle(order)  # orden intercalado distinto por repetición
        for r in order:
            m = models[r]
            acc[r]["fp32"].append(bench(YOLO(str(Path(VOLUME_PATH) / "runs" / r / "weights" / "best.pt")), False))
            acc[r]["fp16"].append(bench(YOLO(str(Path(VOLUME_PATH) / "runs" / r / "weights" / "best.pt")), True))
            for p in photos[:5] + photos[25:30]:
                m.predict(str(p), imgsz=imgsz, device=0, verbose=False)
            for key, subset in (("svc_big", big), ("svc_small", small)):
                ts = []
                for p in subset:
                    t0 = time.perf_counter(); m.predict(str(p), imgsz=imgsz, device=0, verbose=False); ts.append((time.perf_counter() - t0) * 1000)
                acc[r][key].append({"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))})
    for r in run_names:
        res = {k: {"p50_ms_median_of_reps": float(np.median([x["p50_ms"] for x in v])), "p50_ms_reps": [x["p50_ms"] for x in v],
                   "p95_ms_median_of_reps": float(np.median([x["p95_ms"] for x in v]))} for k, v in acc[r].items()}
        out["runs"][r] = res
        print(r, json.dumps(res))
        json.dump(res, open(Path(VOLUME_PATH) / "runs" / r / f"latency_l4_{imgsz}_v2.json", "w"), indent=2)
    volume.commit()
    return out


@app.local_entrypoint()
def main(model: str = "yolo11m.pt", classes: str = "c1", imgsz: int = 640, seed: int = 42, epochs: int = 0,
         models: str = "", seeds: str = "", latency_runs: str = "", fixed: bool = False, fliplr: float = -1.0):
    import json
    """Una corrida (--model) o una tanda (--models "a.pt,b.pt" [--seeds "42,123"]) en paralelo con train.starmap."""
    if latency_runs:
        out = latency.remote([r.strip() for r in latency_runs.split(",") if r.strip()], imgsz)
        print(json.dumps(out, indent=2)); return
    if models or seeds:
        ms = [m.strip() for m in (models or model).split(",") if m.strip()]
        ss = [int(x) for x in (seeds or str(seed)).split(",")]
        fl = None if fliplr < 0 else fliplr
        jobs = [(m, classes, imgsz, sd, epochs or None, fixed, fl) for m in ms for sd in ss]
        print(f"lanzando {len(jobs)} corridas: {[f'{Path(j[0]).stem}_s{j[3]}' for j in jobs]}")
        for meta in train.starmap(jobs):
            pc = meta["metrics"]["val"]["per_class"].get("competidor_number", {})
            print(f"done {meta['run_name']}: {meta['train_seconds']:.0f}s val AP50 {pc.get('AP50'):.4f} AP50:95 {pc.get('AP50_95'):.4f}")
        return
    meta = train.remote(model, classes, imgsz, seed, epochs or None, fixed, None if fliplr < 0 else fliplr)
    print(f"done {meta['run_name']}: val plate AP50 {meta['metrics']['val']['per_class'].get('competidor_number', {}).get('AP50')}")
