"""Terminal-based color labeling tool for the validation set.

Shows each crop in a window, you type the canonical palette name(s).
Progress auto-saves after every label. Resume anytime.

Output: JSONL where each line is {"crop_file", "region", "top1",
"top2"|null, "notes"}.

Controls (text in terminal, after Enter):
    rojo                  → top1 only
    rojo,blanco           → top1 + top2
    rojo,blanco,nota...   → top1, top2, free-form note
    a                     → mark as "acromatico" only (gray-only crop)
    s                     → skip (unlabelable: occluded, ambiguous)
    u                     → undo last label
    q                     → quit (progress saved)
    h                     → list canonical palette names
    n                     → toggle synonym normalization (default ON)

Canonical palette names are accepted; SYNONYM_MAP resolves common
variants (colorado→rojo, lila→morado, café→marron, etc).

Usage:
    uv run python scripts/label_color_crops.py
    uv run python scripts/label_color_crops.py --region helmet
    uv run python scripts/label_color_crops.py --start 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from cycling_photo_ai.color.palette.canonical import PALETTE_NAMES
from cycling_photo_ai.color.palette.synonyms import normalize_query_color
from cycling_photo_ai.shared.paths import COLOR_CROPS_DIR, COLOR_LABELS_DIR

METADATA_CSV = COLOR_CROPS_DIR / "metadata.csv"
LABELS_JSONL = COLOR_LABELS_DIR / "validation.jsonl"

WINDOW_NAME = "Color Labeler — type palette name, 'h' help, 's' skip, 'q' quit"
ACROMATIC = "acromatico"
VALID_NAMES = set(PALETTE_NAMES) | {ACROMATIC}


def load_metadata() -> list[dict]:
    with open(METADATA_CSV) as f:
        return list(csv.DictReader(f))


def load_existing_labels() -> dict[str, dict]:
    if not LABELS_JSONL.exists():
        return {}
    out: dict[str, dict] = {}
    with open(LABELS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["crop_file"]] = entry
    return out


def save_labels(labels: dict[str, dict]) -> None:
    LABELS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_JSONL, "w") as f:
        for entry in labels.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def show_crop(crop_path: Path, idx: int, total: int, region: str) -> None:
    img = cv2.imread(str(crop_path))
    if img is None:
        print(f"  [warn] could not read {crop_path}")
        return
    # Upscale small crops for visibility
    h, w = img.shape[:2]
    target = 400
    if max(h, w) < target:
        scale = target / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    title = f"[{idx + 1}/{total}] {region} — {crop_path.name}"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(WINDOW_NAME, title)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(1)


def parse_input(
    raw: str, normalize: bool
) -> tuple[str, str | None, str | None, str] | str:
    """Return (top1, top2|None, top3|None, notes) or a control char."""
    s = raw.strip()
    if not s:
        return ("", None, None, "")
    if s.lower() in {"s", "u", "q", "h", "n"}:
        return s.lower()
    if s.lower() == "a":
        return (ACROMATIC, None, None, "")

    parts = [p.strip() for p in s.split(",", 3)]
    top1 = parts[0]
    top2 = parts[1] if len(parts) >= 2 and parts[1] else None
    top3 = parts[2] if len(parts) >= 3 and parts[2] else None
    notes = parts[3] if len(parts) >= 4 else ""

    if normalize:
        top1 = normalize_query_color(top1)
        if top2:
            top2 = normalize_query_color(top2)
        if top3:
            top3 = normalize_query_color(top3)

    return (top1, top2, top3, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label color crops for F4 calibration")
    parser.add_argument("--start", type=int, default=0, help="Start from this index")
    parser.add_argument(
        "--region", choices=["helmet", "cyclist_clothes", "bicycle"], default=None,
        help="Filter to a single region (default: all)",
    )
    args = parser.parse_args()

    if not METADATA_CSV.exists():
        sys.exit(f"No metadata at {METADATA_CSV}. Run extract_color_crops.py first.")

    metadata = load_metadata()
    if args.region:
        metadata = [r for r in metadata if r["region"] == args.region]

    labels = load_existing_labels()
    total = len(metadata)
    print(f"Loaded {total} crops, {len(labels)} already labeled")
    print(f"Palette names: {', '.join(PALETTE_NAMES)}")
    print("Type 'h' for help.\n")

    normalize_on = True
    history: list[str] = []
    idx = args.start

    while idx < total:
        row = metadata[idx]
        crop_file = row["crop_file"]
        crop_path = COLOR_CROPS_DIR / crop_file

        if not crop_path.exists():
            print(f"  [skip] missing {crop_path}")
            idx += 1
            continue

        if crop_file in labels:
            idx += 1
            continue

        show_crop(crop_path, idx, total, row["region"])

        prompt = f"[{idx + 1}/{total}] {row['region']} {crop_file} > "
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nQuit. Progress saved.")
            break

        result = parse_input(raw, normalize_on)
        if result == "h":
            print(f"  Palette: {', '.join(PALETTE_NAMES)}")
            print("  Format: top1 | top1,top2 | top1,top2,top3 | top1,top2,top3,notes")
            print("  Controls: a=acromatico, s=skip, u=undo, n=toggle normalize, q=quit, h=help")
            continue
        if result == "n":
            normalize_on = not normalize_on
            print(f"  Synonym normalization: {'ON' if normalize_on else 'OFF'}")
            continue
        if result == "s":
            history.append(crop_file)
            labels[crop_file] = {
                "crop_file": crop_file,
                "region": row["region"],
                "top1": None,
                "top2": None,
                "top3": None,
                "notes": "skipped",
            }
            save_labels(labels)
            idx += 1
            continue
        if result == "u":
            if not history:
                print("  Nothing to undo.")
                continue
            last = history.pop()
            labels.pop(last, None)
            save_labels(labels)
            # Find that crop's index and rewind
            for i, r in enumerate(metadata):
                if r["crop_file"] == last:
                    idx = i
                    break
            print(f"  Undid {last}.")
            continue
        if result == "q":
            print("Quit. Progress saved.")
            break

        top1, top2, top3, notes = result
        if not top1:
            print("  [empty input — try again or 's' to skip]")
            continue
        if top1 not in VALID_NAMES:
            print(f"  [invalid '{top1}'. Type 'h' for valid names. 's' to skip.]")
            continue
        if top2 and top2 not in VALID_NAMES:
            print(f"  [invalid top2 '{top2}'. Try again.]")
            continue
        if top3 and top3 not in VALID_NAMES:
            print(f"  [invalid top3 '{top3}'. Try again.]")
            continue

        labels[crop_file] = {
            "crop_file": crop_file,
            "region": row["region"],
            "top1": top1,
            "top2": top2 if top2 else None,
            "top3": top3 if top3 else None,
            "notes": notes,
        }
        save_labels(labels)
        history.append(crop_file)
        idx += 1

    cv2.destroyAllWindows()

    labeled = sum(1 for e in labels.values() if e.get("top1"))
    skipped = sum(1 for e in labels.values() if e.get("notes") == "skipped")
    print(f"\nLabeled: {labeled} / {total}")
    print(f"Skipped: {skipped}")
    print(f"Output: {LABELS_JSONL}")


if __name__ == "__main__":
    main()
