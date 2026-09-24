"""Barrido D-FINE (tercera familia, D-3): S, M, L a 640, ajuste fino desde pesos COCO con la receta oficial de
fine-tuning de cada variante (configs/dfine/custom/*, Peterande/D-FINE), dataset COCO del split v4 (1 clase).

El repo no es instalable con pip: se clona en la imagen y se fija el commit (se registra en run_meta.json).
Salida por corrida (runs/dfine_<variant>_640_s<seed>/): checkpoints, run_meta.json y dt_val.json / dt_test.json
(detecciones COCO en píxeles originales, umbral 0) para eval_boxes.py --dt-json.

    uv run modal run --detach scripts/eval/modal_train_dfine_sweep.py --models "s" --seeds "42"
    uv run modal run scripts/eval/modal_train_dfine_sweep.py --latency-runs "dfine_s_640_s42,dfine_m_640_s42"
"""

from __future__ import annotations

from pathlib import Path

import modal

GPU_TRAIN = "A100-80GB"   # D-FINE con lote 16 y aumentaciones multiescala superó 38 GB en 40 GB (2026-09-23)
VOL_NAME = "cycling-photo-ai-sweep-vol"
VOLUME_PATH = "/vol"
REPO = "https://github.com/Peterande/D-FINE.git"
WEIGHTS = {v: f"https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{v}_coco.pth" for v in "sml"}

app = modal.App("eswa-dfine-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "faster-coco-eval>=1.6.6", "PyYAML", "tensorboard", "scipy", "calflops",
                 "transformers", "loguru", "pycocotools>=2.0.8", "pillow", "numpy", "matplotlib", "opencv-python-headless")
    .run_commands(f"git clone {REPO} /opt/dfine && cd /opt/dfine && git checkout 956d1709314c2c6a4df6f34de232054578a7449f && git rev-parse HEAD > /opt/dfine/COMMIT",
                  "mkdir -p /opt/weights && cd /opt/weights && " + " && ".join(f"wget -q {u}" for u in WEIGHTS.values()))
    .add_local_file("experiments/split_v4/split.csv", "/root/split.csv")
)

CFG_TEMPLATE = """__include__: ['/opt/dfine/configs/dfine/custom/dfine_hgnetv2_{v}_custom.yml']
output_dir: {out}
num_classes: 2   # ids 0 (placa); margen de 1 por la convención 0-based de nuestro COCO
remap_mscoco_category: False
epochs: 80        # mismo presupuesto de épocas para S, M y L (= receta oficial de L y = RF-DETR); las oficiales de S (220) y M (132) no caben en presupuesto
train_dataloader:
  num_workers: 16
  total_batch_size: 16   # desvío de la receta oficial (S 64, M 32, L 32): 64 no cabe en 40 GB; 16 = lote efectivo de RF-DETR
  collate_fn:
    stop_epoch: 72
  dataset:
    img_folder: {ds}/train
    ann_file: {ds}/train/_annotations.coco.json
    transforms:
      policy:
        epoch: 72
val_dataloader:
  num_workers: 16
  total_batch_size: 16
  dataset:
    img_folder: {ds}/valid
    ann_file: {ds}/valid/_annotations.coco.json
"""

# Resolución distinta de 640 (G3 de D-FINE): se reescriben las tres entradas de tamaño de la receta oficial
# (Resize de train y val, base_size del multiescala, eval_spatial_size); las demás ops son las oficiales.
CFG_RES = """eval_spatial_size: [{r}, {r}]
train_dataloader:
  dataset:
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{r}, {r}]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
  collate_fn:
    base_size: {r}
val_dataloader:
  dataset:
    transforms:
      ops:
        - {{type: Resize, size: [{r}, {r}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
"""
DATA_BY_RES = {640: "v4_c1_coco_800", 800: "v4_c1_coco_1000"}   # copia reducida a 1,25 × la resolución de entrada


@app.function(image=image, gpu=GPU_TRAIN, timeout=6 * 3600, volumes={VOLUME_PATH: volume}, cpu=16, memory=32768)
def train(variant: str, seed: int, res: int = 640) -> dict:
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import time

    import torch

    ds = f"{VOLUME_PATH}/datasets/{DATA_BY_RES[res]}"   # copia reducida (lado mayor 800; D-FINE entrena a 640): D-FINE decodifica en CPU a resolución completa; con 12 Mpx tardaba 14 s por lote
    run_name = f"dfine_{variant}_{res}_s{seed}"
    out = Path(VOLUME_PATH) / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config_eswa.yml"
    cfg_text = CFG_TEMPLATE.format(v=variant, out=str(out), ds=ds)
    if res != 640:   # fusión profunda (un segundo train_dataloader en el mismo yml pisaría el primero entero)
        import yaml
        def merge(a, b):
            for k, v in b.items():
                a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
            return a
        cfg_text = yaml.safe_dump(merge(yaml.safe_load(cfg_text), yaml.safe_load(CFG_RES.format(r=res))), sort_keys=False)
    cfg_path.write_text(cfg_text)
    commit = open("/opt/dfine/COMMIT").read().strip()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"D-FINE {commit[:8]} torch {torch.__version__} GPU {gpu_name} run {run_name}")
    t0 = time.time()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--master_port=7777", "--nproc_per_node=1", "train.py",
           "-c", str(cfg_path), "--use-amp", f"--seed={seed}", "-t", f"/opt/weights/dfine_{variant}_coco.pth",
           "--output-dir", str(out), "-u", "epochs=80"]   # el epochs del yml incluido pisa al de la plantilla; se fuerza aquí
    p = subprocess.run(cmd, cwd="/opt/dfine", env={**os.environ, "PYTHONPATH": "/opt/dfine", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    if p.returncode != 0:
        raise SystemExit(f"train failed rc={p.returncode}")
    train_s = time.time() - t0

    # Mejor checkpoint (stg2 > stg1 > last) → detecciones COCO en val y test
    best = next((out / n for n in ("best_stg2.pth", "best_stg1.pth", "last.pth") if (out / n).exists()), None)
    sys.path.insert(0, "/opt/dfine")
    from src.core import YAMLConfig
    import torchvision.transforms as T
    from PIL import Image
    cfg = YAMLConfig(str(cfg_path), resume=str(best))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ck = torch.load(str(best), map_location="cpu")
    state = ck["ema"]["module"] if "ema" in ck else ck["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy().cuda().eval()
    post = cfg.postprocessor.deploy()
    tf = T.Compose([T.Resize((res, res)), T.ToTensor()])
    ds_full = f"{VOLUME_PATH}/datasets/v4_c1_coco"   # detecciones sobre las fotos originales (píxeles originales, como el resto de familias)
    for split, folder in (("val", "valid"), ("test", "test")):
        paths = sorted(p for p in Path(f"{ds_full}/{folder}").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        dets = []
        with torch.no_grad():
            for i, pth in enumerate(paths):
                im = Image.open(pth).convert("RGB"); W, H = im.size
                x = tf(im).unsqueeze(0).cuda(); sz = torch.tensor([[W, H]]).cuda()
                labels, boxes, scores = post(model(x), sz)
                for l, b, s in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                    x1, y1, x2, y2 = b
                    dets.append({"image_id": i, "category_id": int(l), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        json.dump(dets, open(out / f"dt_{split}.json", "w"))
        print(f"{split}: {len(paths)} imgs, {len(dets)} dets")
    meta = {"run_name": run_name, "family": "dfine", "variant": variant, "resolution": res, "seed": seed, "commit": commit,
            "torch": torch.__version__, "gpu": gpu_name, "train_seconds": round(train_s, 1), "best_checkpoint": best.name,
            "pretrain": WEIGHTS[variant], "params_M": round(sum(p.numel() for p in cfg.model.parameters()) / 1e6, 3),
            "split_sha256": hashlib.sha256(open("/root/split.csv", "rb").read()).hexdigest(),
            "recipe": f"configs/dfine/custom/dfine_hgnetv2_{variant}_custom.yml (oficial) con total_batch_size 16, epochs 80 (policy 72), datos a {DATA_BY_RES[res]}, entrada {res}, -t coco, --use-amp"}
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2)
    volume.commit()
    print(json.dumps(meta, indent=2))
    return meta


BENCH_ONE = r"""
import json, sys, time, numpy as np, torch, torchvision.transforms as T
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/opt/dfine")
from src.core import YAMLConfig
run, vol = sys.argv[1], sys.argv[2]
out = Path(vol) / "runs" / run
meta = json.load(open(out / "run_meta.json"))
cfg = YAMLConfig(str(out / "config_eswa.yml"), resume=str(out / meta["best_checkpoint"]))
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
ck = torch.load(str(out / meta["best_checkpoint"]), map_location="cpu")
cfg.model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
net = cfg.model.deploy().cuda().eval(); post = cfg.postprocessor.deploy()
def bench(half):
    x = torch.rand(1, 3, meta["resolution"], meta["resolution"], device="cuda")
    ctx = torch.autocast("cuda", dtype=torch.float16) if half else torch.no_grad()
    with torch.no_grad(), ctx:
        for _ in range(20): net(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(100):
            t0 = time.perf_counter(); net(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts); return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}
tf = T.Compose([T.Resize((meta["resolution"], meta["resolution"])), T.ToTensor()])
allp = sorted((Path(vol) / "datasets" / "v4_c1_coco" / "test").glob("*.jpg"))
big = [p for p in allp if max(Image.open(p).size) >= 2000][:25]; small = [p for p in allp if max(Image.open(p).size) < 2000][:25]
def svc(p):
    im = Image.open(p).convert("RGB"); W, H = im.size
    with torch.no_grad(): post(net(tf(im).unsqueeze(0).cuda()), torch.tensor([[W, H]]).cuda())
res = {"fp32": bench(False), "fp16_autocast": bench(True)}
for p in big[:5] + small[:5]: svc(p)
for key, subset in (("svc_big", big), ("svc_small", small)):
    ts = []
    for p in subset:
        t0 = time.perf_counter(); svc(p); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    res[key] = {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
print("RESULT " + json.dumps(res))
"""


@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume}, cpu=4, memory=16384)
def latency(run_names: list[str]) -> dict:
    """Protocolo v2 (un contenedor, orden intercalado, 3 repeticiones, mediana). Cada medición corre en un subproceso
    limpio porque el registro global de configuración de D-FINE se contamina entre variantes en el mismo proceso."""
    import json
    import random
    import subprocess
    import sys

    import numpy as np
    import torch

    acc = {r: {"fp32": [], "fp16_autocast": [], "svc_big": [], "svc_small": []} for r in run_names}
    order = list(run_names)
    for rep in range(3):
        random.Random(rep).shuffle(order)
        for r in order:
            p = subprocess.run([sys.executable, "-c", BENCH_ONE, r, VOLUME_PATH], capture_output=True, text=True, cwd="/opt/dfine")
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                raise SystemExit(f"bench failed for {r}: {p.stderr[-2000:]}")
            res = json.loads(line[-1][7:])
            for k in acc[r]:
                acc[r][k].append(res[k])
    out = {"gpu": torch.cuda.get_device_name(0), "reps": 3, "runs": {}}
    for r in run_names:
        res = {k: {"p50_ms_median_of_reps": float(np.median([x["p50_ms"] for x in v])), "p50_ms_reps": [x["p50_ms"] for x in v],
                   "p95_ms_median_of_reps": float(np.median([x["p95_ms"] for x in v]))} for k, v in acc[r].items()}
        out["runs"][r] = res
        json.dump(res, open(Path(VOLUME_PATH) / "runs" / r / "latency_l4_v2.json", "w"), indent=2)
        print(r, json.dumps(res))
    volume.commit()
    return out


@app.local_entrypoint()
def main(models: str = "s", seeds: str = "42", latency_runs: str = "", res: int = 640):
    if latency_runs:
        latency.remote([r.strip() for r in latency_runs.split(",") if r.strip()]); return
    jobs = [(m.strip(), int(s), res) for m in models.split(",") for s in seeds.split(",")]
    print(f"lanzando {len(jobs)}: {[f'dfine_{j[0]}_{res}_s{j[1]}' for j in jobs]}")
    for meta in train.starmap(jobs):
        print(f"done {meta['run_name']}: {meta['train_seconds']:.0f}s")
