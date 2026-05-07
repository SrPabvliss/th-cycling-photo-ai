"""Visual end-to-end demo — detection + OCR + color overlay.

Production pipeline (YOLO11m v3_cleaned + PARSeq 4-phase + ManualColor S3
final) on a real image. Renders detection bboxes, bib OCR, and per-region
dominant color into a single matplotlib figure.

Usage:
    uv run python demo_visual.py                          # random v3_cleaned valid image
    uv run python demo_visual.py path/to/image.jpg        # specific image
    uv run python demo_visual.py path/to/image.jpg --color gemini  # use Gemini color (network call)
    uv run python demo_visual.py --no-color               # skip color
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMG_DIR = PROJECT_ROOT / "experiments" / "audit_adr015" / "dataset" / "v3_cleaned" / "valid"

CLASS_COLORS = {
    "bicycle": "#2196F3",
    "competidor_number": "#FF5722",
    "cyclist": "#4CAF50",
    "cyclist_clothes": "#9C27B0",
    "cyclist_with_bike": "#FF9800",
    "helmet": "#00BCD4",
}

# Approximate display swatches per palette name (EN). Used only for the
# legend chip on top of each color region — not sent to/from the analyzer.
EN_SWATCH = {
    "black": "#000000", "white": "#FFFFFF", "gray": "#808080",
    "red": "#D32F2F", "orange": "#FB8C00", "yellow": "#FBC02D",
    "green": "#388E3C", "blue": "#1976D2", "cyan": "#00ACC1",
    "purple": "#7B1FA2", "pink": "#EC407A", "magenta": "#D81B60",
    "brown": "#6D4C41", "silver": "#BDBDBD", "gold": "#C9A227",
}


def pick_image(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            raise SystemExit(f"Image not found: {p}")
        return p
    candidates = sorted(DEFAULT_IMG_DIR.glob("*.jpg"))
    if not candidates:
        raise SystemExit(f"No images in {DEFAULT_IMG_DIR}")
    return random.choice(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("--color", choices=["manual", "gemini", "none"], default="manual")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    color_type = "none" if args.no_color else args.color

    img_path = pick_image(args.image)
    print(f"Image: {img_path}")
    print(f"Color strategy: {color_type}\n")

    from cycling_photo_ai.pipeline.app import _get_orchestrator

    orch = _get_orchestrator("yolo", "parseq", color_type)
    result = orch.process(str(img_path))

    print(f"Detections:     {len(result.detections)}")
    print(f"Bib readings:   {len(result.bib_readings)}")
    print(f"Color analyses: {len(result.color_analyses)}")
    print(f"Processing ms:  {result.processing_ms:.0f}\n")

    image_bgr = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image_rgb)
    ax.set_title(
        f"{img_path.name}  •  detections={len(result.detections)}  "
        f"bibs={len(result.bib_readings)}  colors={len(result.color_analyses)}  "
        f"({result.processing_ms:.0f}ms)",
        fontsize=11,
    )
    ax.axis("off")

    # Detection bboxes
    color_by_bbox: dict[tuple[float, ...], dict] = {
        tuple(c["bbox_source"]): c for c in result.color_analyses
    }
    bib_by_bbox: dict[tuple[float, ...], dict] = {
        tuple(b["bbox_source"]): b for b in result.bib_readings
    }

    for det in result.detections:
        cls_name = det["class_name"]
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        bw, bh = x2 - x1, y2 - y1
        edge = CLASS_COLORS.get(cls_name, "#888888")
        lw = 3 if cls_name == "competidor_number" else 1.5
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=lw, edgecolor=edge, facecolor="none")
        ax.add_patch(rect)
        label = f"{cls_name} {det['confidence']:.2f}"
        ax.text(x1, y1 - 4, label, fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=edge, alpha=0.85))

        # OCR overlay
        bib = bib_by_bbox.get(tuple(det["bbox"]))
        if bib:
            digits = bib["digits"] or "???"
            conf = bib["confidence"]
            sc = "#4CAF50" if conf >= 0.7 else "#FF9800" if conf >= 0.4 else "#F44336"
            ax.text(x1 + bw / 2, y2 + 16, f"#{digits} ({conf:.2f})",
                    fontsize=11, fontweight="bold", color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=sc, alpha=0.9))

        # Color overlay
        col = color_by_bbox.get(tuple(det["bbox"]))
        if col:
            primary = col["primary_color"]
            secondary = col["secondary_color"]
            swatch_main = EN_SWATCH.get(primary, "#888888")
            label_txt = primary if not secondary else f"{primary}+{secondary}"
            ax.text(
                x2 - 4, y1 + 12,
                label_txt, fontsize=8, color="black", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=swatch_main,
                          edgecolor="white", linewidth=1, alpha=0.95),
            )

    # Legend
    handles = [patches.Patch(facecolor=c, label=n) for n, c in CLASS_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    # Footer text
    if result.color_analyses:
        lines = []
        for c in result.color_analyses:
            lines.append(
                f"  {c['region']:<18} {c['primary_color']:<10} "
                f"+{str(c['secondary_color']):<10} conf={c['confidence']:.2f} "
                f"({c['strategy']}, {c['processing_ms']}ms)"
            )
        print("\n".join(lines))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
