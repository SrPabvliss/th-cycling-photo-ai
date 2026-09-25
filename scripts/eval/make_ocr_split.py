"""Q-11: group-aware, stratified split of the 655 labeled bib crops (OCR sweep, ESWA article).

Why: the thesis split (`data/ocr/dataset`, fold_0 + 99-crop test) was stratified by digit
length only. The same rider appears in many photos, so 81 of the 99 test crops share their
bib number with train+val. This script groups crops by bib number AND by source photo
(union-find), assigns whole groups to one split, stratified by digit length, and freezes
the result with SHA-256 per crop. It also records the detector split (v4) of the source
photo, for later crop-stability tests.

Usage:
    uv run python scripts/eval/make_ocr_split.py --out experiments/ocr_split_v2 --seed 42

Outputs (in --out):
    split.csv     crop_file, bib_number, n_digits, source_image, split_old, thesis_test,
                  split_v4, split, group_id, group_size, width, height, sha256
    summary.json  counts per split and stratum, leakage checks, parameters
    lmdb/{train,val,test}   LMDB in the same format as data/ocr/dataset (PNG, original size)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data" / "ocr" / "crops"
LABELS = CROPS / "labels.csv"
THESIS_TEST = ROOT / "data" / "ocr" / "dataset" / "test" / "manifest.json"
SPLIT_V4 = ROOT / "experiments" / "split_v4" / "split.csv"
SPLITS = ["train", "val", "test"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def write_lmdb(path: Path, rows: list[dict]) -> None:
    import lmdb

    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=max(len(rows) * 400 * 1024, 32 << 20))
    with env.begin(write=True) as txn:
        for i, r in enumerate(rows):
            img = Image.open(CROPS / r["crop_file"]).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            txn.put(f"image-{i+1:09d}".encode(), buf.getvalue())
            txn.put(f"label-{i+1:09d}".encode(), r["bib_number"].encode())
        txn.put(b"num-samples", str(len(rows)).encode())
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/ocr_split_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", default="0.70,0.15,0.15")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fracs = dict(zip(SPLITS, map(float, args.frac.split(","))))
    rng = random.Random(args.seed)

    rows = [r for r in csv.DictReader(open(LABELS)) if r["bib_number"] != "SKIP"]
    thesis_test = {s["crop_file"] for s in json.load(open(THESIS_TEST))["samples"]}
    v4 = {r["file_name"]: r["split"] for r in csv.DictReader(open(SPLIT_V4))} if SPLIT_V4.exists() else {}

    # ---- groups: same bib number OR same source photo ----
    uf = UF()
    for r in rows:
        uf.union(("bib", r["bib_number"]), ("img", r["source_image"]))
    groups: dict = defaultdict(list)
    for r in rows:
        groups[uf.find(("bib", r["bib_number"]))].append(r)
    glist = list(groups.values())
    rng.shuffle(glist)
    glist.sort(key=len, reverse=True)  # big groups first, then random order among equals

    # ---- stratified greedy assignment by dominant digit length ----
    def stratum(g):
        return Counter(len(r["bib_number"]) for r in g).most_common(1)[0][0]

    total_by_s = Counter()
    for g in glist:
        total_by_s[stratum(g)] += len(g)
    target = {(s, k): fracs[k] * n for s, n in total_by_s.items() for k in SPLITS}
    filled = Counter()
    assign = {}
    for gi, g in enumerate(glist):
        s = stratum(g)
        # split with the largest remaining deficit in this stratum (ties: fixed order)
        k = max(SPLITS, key=lambda k: (target[(s, k)] - filled[(s, k)]) / max(fracs[k], 1e-9))
        for r in g:
            assign[r["crop_file"]] = (k, gi, len(g))
        filled[(s, k)] += len(g)

    # ---- write split.csv ----
    fields = ["crop_file", "bib_number", "n_digits", "source_image", "split_old", "thesis_test",
              "split_v4", "split", "group_id", "group_size", "width", "height", "sha256"]
    out_rows = []
    for r in rows:
        k, gi, gs = assign[r["crop_file"]]
        w, h = Image.open(CROPS / r["crop_file"]).size
        out_rows.append({
            "crop_file": r["crop_file"], "bib_number": r["bib_number"], "n_digits": len(r["bib_number"]),
            "source_image": r["source_image"], "split_old": r["source_split"],
            "thesis_test": int(r["crop_file"] in thesis_test), "split_v4": v4.get(r["source_image"], ""),
            "split": k, "group_id": gi, "group_size": gs, "width": w, "height": h,
            "sha256": sha256(CROPS / r["crop_file"]),
        })
    out_rows.sort(key=lambda r: r["crop_file"])
    with open(out / "split.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out_rows)

    # ---- checks ----
    by_split = defaultdict(list)
    for r in out_rows:
        by_split[r["split"]].append(r)
    bib_splits = defaultdict(set)
    img_splits = defaultdict(set)
    for r in out_rows:
        bib_splits[r["bib_number"]].add(r["split"])
        img_splits[r["source_image"]].add(r["split"])
    leak_bib = sum(len(v) > 1 for v in bib_splits.values())
    leak_img = sum(len(v) > 1 for v in img_splits.values())
    assert leak_bib == 0 and leak_img == 0, (leak_bib, leak_img)
    assert len({r["sha256"] for r in out_rows}) == len(out_rows), "duplicate crop bytes"

    summary = {
        "seed": args.seed, "fracs": fracs, "n_crops": len(out_rows), "n_groups": len(glist),
        "group_size_max": max(len(g) for g in glist),
        "counts": {k: len(v) for k, v in by_split.items()},
        "by_digits": {k: dict(Counter(r["n_digits"] for r in v)) for k, v in by_split.items()},
        "distinct_bibs": {k: len({r["bib_number"] for r in v}) for k, v in by_split.items()},
        "thesis_test_in": {k: sum(r["thesis_test"] for r in v) for k, v in by_split.items()},
        "split_v4_of_sources": {k: dict(Counter(r["split_v4"] for r in v)) for k, v in by_split.items()},
        "leak_bib_groups": leak_bib, "leak_source_images": leak_img,
        "height_px": {k: {"min": min(r["height"] for r in v), "median": sorted(r["height"] for r in v)[len(v) // 2]}
                       for k, v in by_split.items()},
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    summary["split_csv_sha256"] = sha256(out / "split.csv")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)

    for k in SPLITS:
        write_lmdb(out / "lmdb" / k, by_split[k])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
