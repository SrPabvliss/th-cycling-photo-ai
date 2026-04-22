"""Terminal-based bib number labeling tool.

Shows each crop in a window, you type the number and press Enter.
Progress auto-saves after every label. Resume anytime.

Controls:
    [number] + Enter  → label this crop with that number
    s + Enter         → skip (illegible/unclear)
    u + Enter         → undo last label
    q + Enter         → quit (progress saved)

Usage:
    uv run python scripts/label_bib_crops.py
    uv run python scripts/label_bib_crops.py --start 500   # resume from crop 500
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
METADATA_CSV = CROPS_DIR / "crops_metadata.csv"
LABELS_CSV = CROPS_DIR / "labels.csv"

WINDOW_NAME = "Bib Labeler — type number, Enter to confirm, 's' skip, 'q' quit"


def load_existing_labels() -> dict[str, str]:
    """Load already-labeled crops."""
    if not LABELS_CSV.exists():
        return {}

    labels = {}
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["crop_file"]] = row["bib_number"]
    return labels


def save_labels(labels: dict[str, str], metadata: list[dict]):
    """Save labels to CSV."""
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_file", "bib_number", "source_image", "source_split"])
        writer.writeheader()
        for row in metadata:
            if row["crop_file"] in labels:
                writer.writerow({
                    "crop_file": row["crop_file"],
                    "bib_number": labels[row["crop_file"]],
                    "source_image": row["source_image"],
                    "source_split": row["source_split"],
                })


def load_metadata() -> list[dict]:
    """Load crops metadata."""
    with open(METADATA_CSV) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Label bib number crops")
    parser.add_argument("--start", type=int, default=0, help="Start from this crop index")
    args = parser.parse_args()

    if not METADATA_CSV.exists():
        print(f"No metadata found. Run extract_bib_crops.py first.")
        sys.exit(1)

    metadata = load_metadata()
    labels = load_existing_labels()
    total = len(metadata)
    labeled_count = len(labels)

    print(f"Total crops: {total}")
    print(f"Already labeled: {labeled_count}")
    print(f"Remaining: {total - labeled_count}")
    print(f"\nControls: [number]+Enter=label, s=skip, u=undo, q=quit")
    print(f"{'='*60}\n")

    # Find first unlabeled crop (or start from --start)
    start_idx = args.start
    history: list[str] = []  # for undo

    i = start_idx
    while i < total:
        row = metadata[i]
        crop_file = row["crop_file"]

        # Skip already labeled
        if crop_file in labels and i >= start_idx:
            i += 1
            continue

        crop_path = CROPS_DIR / crop_file
        if not crop_path.exists():
            i += 1
            continue

        # Show crop
        img = cv2.imread(str(crop_path))
        if img is None:
            i += 1
            continue

        # Resize for visibility (min 200px height)
        h, w = img.shape[:2]
        if h < 200:
            scale = 200 / h
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(1)  # refresh window

        # Stats
        done = len(labels)
        pct = done / total * 100
        remaining = total - done

        prompt = f"[{i+1}/{total}] ({pct:.0f}% done, {remaining} left) {crop_file} → bib number: "

        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and quitting...")
            break

        if user_input.lower() == "q":
            break
        elif user_input.lower() == "s":
            labels[crop_file] = "SKIP"
            history.append(crop_file)
            i += 1
        elif user_input.lower() == "u":
            if history:
                last = history.pop()
                if last in labels:
                    del labels[last]
                    print(f"  Undid label for {last}")
                    # Go back to that crop
                    for j, r in enumerate(metadata):
                        if r["crop_file"] == last:
                            i = j
                            break
            else:
                print("  Nothing to undo")
        elif user_input.isdigit() and 0 < len(user_input) <= 4:
            labels[crop_file] = user_input
            history.append(crop_file)
            i += 1

            # Auto-save every 50 labels
            if len(labels) % 50 == 0:
                save_labels(labels, metadata)
                print(f"  [auto-saved at {len(labels)} labels]")
        else:
            print(f"  Invalid: type 1-4 digits, 's' to skip, 'q' to quit")

    # Final save
    cv2.destroyAllWindows()
    save_labels(labels, metadata)

    labeled = sum(1 for v in labels.values() if v != "SKIP")
    skipped = sum(1 for v in labels.values() if v == "SKIP")
    print(f"\n{'='*60}")
    print(f"Saved {labeled} labels + {skipped} skips to {LABELS_CSV}")
    print(f"Remaining: {total - len(labels)}")

    if labeled > 0:
        # Quick stats
        lengths = {}
        for v in labels.values():
            if v != "SKIP":
                l = len(v)
                lengths[l] = lengths.get(l, 0) + 1
        print(f"\nDigit length distribution:")
        for l in sorted(lengths):
            print(f"  {l} digits: {lengths[l]} ({lengths[l]/labeled*100:.0f}%)")


if __name__ == "__main__":
    main()
