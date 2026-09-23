"""R-2: group-aware, stratified 70/15/15 split of the 1 200-image dataset (v3_cleaned).

Why: the Roboflow split (903/198/99) is random per image. Downhill photos come in
bursts of the same rider on the same run, so near-identical frames landed on both
sides of the split (ADR-015 audit: 14 cross-split near-duplicates). This script
groups images of the same burst (same rider, same run). Rule, chosen after
profiling (see summary.json): two images are linked if they come from the same
camera with frame numbers at most 3 apart AND their DINOv2-small embeddings are
within cosine distance 0.10 (consecutive AND similar = same rider), or if their
embeddings are within 0.05 regardless of numbering (near-duplicates, needed for
the 300 images without frame numbers). Pure embedding single-linkage at 0.10
chained 239 different photos of one event into one group; frame adjacency alone
chained long sequences of different riders. Groups are unions of those links and assigns whole groups to one split, stratified by plate presence,
image-size bucket and camera prefix. The output split.csv (with SHA-256 per file)
is the single definition of the split for every detector sweep (ESWA article).

Usage:
    uv run python scripts/eval/make_split.py \
        --src experiments/audit_adr015/dataset/v3_cleaned \
        --out experiments/split_v4 --seed 42

Outputs (in --out):
    split.csv      file_name, orig_name, split_old, split, group_id, group_size,
                   width, height, size_bucket, prefix, n_plates, n_anns, sha256
    summary.json   counts and strata proportions per split, parameters
    embeddings.npy / embeddings_files.json  (cached DINOv2 embeddings)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS_OLD = ["train", "valid", "test"]
PLATE = "competidor_number"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_of(orig_name: str) -> str:
    m = re.match(r"([A-Za-z_]+?)[_\-]?\d", orig_name)
    p = m.group(1) if m else "other"
    return p if p in {"DSC", "_MG", "P", "sultana", "el_calvario"} else "other"


def size_bucket(w: int, h: int) -> str:
    return "large" if max(w, h) >= 2000 else "small"


def load_images(src: Path) -> list[dict]:
    rows = []
    for split in SPLITS_OLD:
        coco = json.load(open(src / split / "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(cats[a["category_id"]])
        for im in coco["images"]:
            names = anns.get(im["id"], [])
            orig = im.get("extra", {}).get("name", im["file_name"])
            rows.append({
                "file_name": im["file_name"], "orig_name": orig, "split_old": split,
                "path": str(src / split / im["file_name"]),
                "width": im["width"], "height": im["height"],
                "size_bucket": size_bucket(im["width"], im["height"]),
                "prefix": prefix_of(orig),
                "n_plates": sum(n == PLATE for n in names), "n_anns": len(names),
            })
    return rows


def embed(rows: list[dict], out: Path) -> np.ndarray:
    cache, cache_files = out / "embeddings.npy", out / "embeddings_files.json"
    files = [r["file_name"] for r in rows]
    if cache.exists() and cache_files.exists() and json.load(open(cache_files)) == files:
        return np.load(cache)
    import torch
    from PIL import Image, ImageOps
    import torchvision.transforms as T
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    embs = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [tf(ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB")) for r in rows[i:i + 32]]
            e = model(torch.stack(batch).to(dev)).float().cpu().numpy()
            embs.append(e)
            print(f"  embedded {min(i + 32, len(rows))}/{len(rows)}", flush=True)
    E = np.concatenate(embs)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    np.save(cache, E); json.dump(files, open(cache_files, "w"))
    return E


def frame_key(orig_name: str):
    m = re.match(r"(DSC|_MG_|P|sultana_|el_calvario_)(\d+)", orig_name)
    return (m.group(1), int(m.group(2))) if m else None


def group(E: np.ndarray, rows: list[dict], thr_adj: float, gap: int, thr_pure: float) -> list[int]:
    """Union-find over: (same camera, |frame diff| <= gap, cosine dist < thr_adj) OR (cosine dist < thr_pure)."""
    n = len(E)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    D = 1 - E @ E.T
    for i, j in zip(*np.where(np.triu(D < thr_pure, k=1))):
        parent[find(i)] = find(j)
    by_cam = defaultdict(list)
    for i, r in enumerate(rows):
        k = frame_key(r["orig_name"])
        if k:
            by_cam[k[0]].append((k[1], i))
    for lst in by_cam.values():
        lst.sort()
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if lst[b][0] - lst[a][0] > gap:
                    break
                i, j = lst[a][1], lst[b][1]
                if D[i, j] < thr_adj:
                    parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def assign(rows, groups, fracs, seed):
    """Greedy stratified assignment of whole groups. Largest groups first; each
    group goes to the split whose current fill (images and per-stratum counts)
    is furthest below target. Deterministic given seed."""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    order = sorted(by_g.values(), key=lambda m: (-len(m), rng.random()))
    total = len(rows)
    strata = ["size_bucket", "prefix", "has_plate"]
    for r in rows:
        r["has_plate"] = "plate" if r["n_plates"] > 0 else "noplate"
    tot_s = {s: Counter(r[s] for r in rows) for s in strata}
    fill = {k: 0 for k in fracs}
    fill_s = {k: {s: Counter() for s in strata} for k in fracs}
    out = [None] * total
    for members in order:
        best, best_score = None, None
        for k, f in fracs.items():
            score = (fill[k] + len(members)) / (f * total)
            for s in strata:
                for i in members:
                    v = rows[i][s]
                    score += 0.25 * (fill_s[k][s][v] + 1) / (f * max(tot_s[s][v], 1)) / len(members)
            if best_score is None or score < best_score - 1e-12 or (abs(score - best_score) < 1e-12 and rng.random() < 0.5):
                best, best_score = k, score
        for i in members:
            out[i] = best
            fill[best] += 1
            for s in strata:
                fill_s[best][s][rows[i][s]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/audit_adr015/dataset/v3_cleaned")
    ap.add_argument("--out", default="experiments/split_v4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr-adj", type=float, default=0.10, help="cosine distance for consecutive-frame links")
    ap.add_argument("--gap", type=int, default=3, help="max frame-number gap for consecutive-frame links")
    ap.add_argument("--thr-pure", type=float, default=0.05, help="cosine distance for unconditional near-duplicate links")
    ap.add_argument("--fracs", default="train:0.70,val:0.15,test:0.15")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fracs = {k: float(v) for k, v in (x.split(":") for x in a.fracs.split(","))}

    rows = load_images(src)
    print(f"{len(rows)} images")
    E = embed(rows, out)
    groups = group(E, rows, a.thr_adj, a.gap, a.thr_pure)
    sizes = Counter(Counter(groups).values())
    print(f"groups: {len(set(groups))}  size histogram: {dict(sorted(sizes.items()))}")
    split = assign(rows, groups, fracs, a.seed)
    gsize = Counter(groups)

    import csv
    cols = ["file_name", "orig_name", "split_old", "split", "group_id", "group_size", "width", "height",
            "size_bucket", "prefix", "n_plates", "n_anns", "sha256"]
    with open(out / "split.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r, g, s in sorted(zip(rows, groups, split), key=lambda x: x[0]["file_name"]):
            w.writerow([r["file_name"], r["orig_name"], r["split_old"], s, g, gsize[g], r["width"], r["height"],
                        r["size_bucket"], r["prefix"], r["n_plates"], r["n_anns"], sha256(Path(r["path"]))])

    summary = {"seed": a.seed, "rule": {"thr_adj": a.thr_adj, "gap": a.gap, "thr_pure": a.thr_pure}, "fracs": fracs, "n_images": len(rows),
               "n_groups": len(set(groups)), "group_size_hist": {str(k): v for k, v in sorted(sizes.items())},
               "per_split": {}}
    for k in fracs:
        sel = [r for r, s in zip(rows, split) if s == k]
        summary["per_split"][k] = {
            "images": len(sel), "plates": sum(r["n_plates"] for r in sel),
            "images_with_plate": sum(r["n_plates"] > 0 for r in sel), "anns": sum(r["n_anns"] for r in sel),
            "size_bucket": dict(Counter(r["size_bucket"] for r in sel)),
            "prefix": dict(Counter(r["prefix"] for r in sel)),
            "from_old_split": dict(Counter(r["split_old"] for r in sel)),
        }
    # leak checks: no link of either kind may cross splits (0 by construction); report nearest cross-split distance
    D = 1 - E @ E.T
    pure = [(i, j) for i, j in zip(*np.where(np.triu(D < a.thr_pure, k=1)))]
    summary["cross_split_pure_pairs"] = int(sum(split[i] != split[j] for i, j in pure))
    adj_cross = 0
    for r_i, i in [(rows[i], i) for i in range(len(rows))]:
        pass
    by_cam = defaultdict(list)
    for i, r in enumerate(rows):
        k = frame_key(r["orig_name"])
        if k:
            by_cam[k[0]].append((k[1], i))
    for lst in by_cam.values():
        lst.sort()
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                if lst[y][0] - lst[x][0] > a.gap:
                    break
                i, j = lst[x][1], lst[y][1]
                if D[i, j] < a.thr_adj and split[i] != split[j]:
                    adj_cross += 1
    summary["cross_split_adjacent_similar_pairs"] = adj_cross
    cross = [(i, j) for i in range(len(E)) for j in range(i + 1, len(E)) if split[i] != split[j]]
    summary["cross_split_min_distance"] = float(min(D[i, j] for i, j in cross))
    summary["group_size_max"] = max(gsize.values())
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
