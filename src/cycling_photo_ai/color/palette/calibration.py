"""Empirical palette centroid calibration from labeled validation data.

For each chromatic palette name (rojo, naranja, ..., dorado), pull the
chromatic pixels from every crop where Pablo labeled top1 == name and
compute their median CIELAB. Median over mean is more robust to outlier
crops (e.g. mis-labeled or compound-color crops where the dominant chroma
isn't actually the labeled color).

Achromatic entries (negro, gris, blanco, plateado) are NOT calibrated —
their canonical centroids are fixed by definition (see
palette_specification.md §calibracion).

The output is a YAML file consumable by the analyzer via PaletteConfig
or PALETTE_LAB monkey-patching at runtime.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.color import rgb2lab

from cycling_photo_ai.color.dataset.validation_set import ValidationCrop, load_validation_set
from cycling_photo_ai.color.inference.pipeline_stages import gray_world

# Names whose centroids are calibrated empirically. Achromatics + dorado/plateado
# (metallic finishes — empirical signature is unstable from a small set) keep
# canonical centroids.
CALIBRATABLE_NAMES = (
    "rojo", "naranja", "amarillo", "verde",
    "azul", "celeste", "morado", "rosa",
    "fucsia", "marron",
)

# Minimum chromatic pixels per name for calibration to be reliable.
MIN_PIXELS_PER_NAME = 200

# Achromatic-side cap: pixels with chroma below this are considered
# achromatic for calibration (kept aligned with the partition default).
CHROMA_FLOOR = 10.0


def collect_anchored_centroids_by_label(
    crops: list[ValidationCrop],
    apply_gray_world: bool = True,
    max_anchor_de: float = 30.0,
) -> dict[str, list[tuple[np.ndarray, int]]]:
    """For each labeled crop, K-Means the chromatic pool and keep the cluster
    CLOSEST to the canonical palette centroid for that label (anchored).

    Strategy: a "rojo bicycle" crop typically has TWO chromatic populations:
    the bicycle frame (rojo) and the background (ground/brush/sky). The
    background often dominates by pixel count, so the largest cluster is
    not always the labeled color. Anchoring to canonical rojo via ΔE_00
    < max_anchor_de selects the cluster that actually represents the
    labeled color, filtering out background.

    Returns:
        {label_name: [(centroid_lab, n_pixels_in_cluster), ...]}
    """
    from sklearn.cluster import KMeans
    from skimage.color import deltaE_ciede2000

    from cycling_photo_ai.color.palette.canonical import PALETTE_LAB

    by_name: dict[str, list[tuple[np.ndarray, int]]] = {}
    K_PER_CROP = 5

    for crop in crops:
        if crop.top1 not in CALIBRATABLE_NAMES:
            continue
        try:
            img = crop.load_bgr()
        except FileNotFoundError:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if apply_gray_world:
            rgb = gray_world(rgb)
        lab = rgb2lab(rgb / 255.0).reshape(-1, 3)

        chroma = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        chromatic = lab[chroma >= CHROMA_FLOOR]
        if chromatic.shape[0] < 200:
            continue

        if chromatic.shape[0] > 20_000:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(chromatic.shape[0], size=20_000, replace=False)
            chromatic = chromatic[idx]

        k = min(K_PER_CROP, chromatic.shape[0])
        km = KMeans(n_clusters=k, init="k-means++", n_init=5,
                    max_iter=100, random_state=42)
        km.fit(chromatic)
        counts = np.bincount(km.labels_, minlength=k)

        # Pick cluster nearest to canonical centroid for this label
        anchor = PALETTE_LAB[crop.top1]
        des = deltaE_ciede2000(
            km.cluster_centers_.reshape(1, -1, 3),
            anchor.reshape(1, 1, 3),
        )[0, :, 0] if False else None
        # The reshape above is fragile across skimage versions; compute pair-wise:
        des = np.array([
            float(deltaE_ciede2000(
                c.reshape(1, 1, 3),
                anchor.reshape(1, 1, 3),
            )[0, 0])
            for c in km.cluster_centers_
        ])

        # Select clusters within max_anchor_de of canonical; pick the largest
        within = np.where(des < max_anchor_de)[0]
        if within.size == 0:
            continue   # no cluster close to canonical → skip this crop
        # Among acceptable clusters, take the largest by pixel count
        chosen = within[np.argmax(counts[within])]
        by_name.setdefault(crop.top1, []).append(
            (km.cluster_centers_[chosen].astype(np.float64), int(counts[chosen]))
        )

    return by_name


def calibrate_centroids(
    crops: list[ValidationCrop],
    apply_gray_world: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Compute empirical centroids per chromatic palette entry."""
    from cycling_photo_ai.color.palette.canonical import PALETTE_LAB

    pools = collect_anchored_centroids_by_label(crops, apply_gray_world=apply_gray_world)

    centroids: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}

    for name in CALIBRATABLE_NAMES:
        contribs = pools.get(name, [])
        n_crops = sum(1 for c in crops if c.top1 == name)

        # Need at least 3 crops to make a stable empirical estimate.
        if len(contribs) < 3:
            centroids[name] = PALETTE_LAB[name].copy()
            stats[name] = {
                "n_pixels": int(sum(s for _, s in contribs)),
                "n_crops": n_crops,
                "n_contributing_crops": len(contribs),
                "calibrated": False,
                "fallback_reason": "<3 contributing crops — using canonical centroid",
            }
            continue

        # Weighted median (per-channel) over crop dominant centroids
        cents = np.stack([c for c, _ in contribs], axis=0)
        weights = np.array([s for _, s in contribs], dtype=np.float64)
        emp_lab = _weighted_median(cents, weights)

        centroids[name] = emp_lab.astype(np.float64)
        stats[name] = {
            "n_pixels": int(sum(s for _, s in contribs)),
            "n_crops": n_crops,
            "n_contributing_crops": len(contribs),
            "calibrated": True,
            "canonical_lab": PALETTE_LAB[name].tolist(),
            "empirical_lab": emp_lab.tolist(),
        }

    return centroids, stats


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-channel weighted median over a (N, 3) value array."""
    out = np.empty(values.shape[1], dtype=np.float64)
    for c in range(values.shape[1]):
        order = np.argsort(values[:, c])
        sorted_vals = values[order, c]
        sorted_w = weights[order]
        cumw = np.cumsum(sorted_w)
        half = sorted_w.sum() / 2.0
        idx = int(np.searchsorted(cumw, half))
        idx = min(idx, len(sorted_vals) - 1)
        out[c] = sorted_vals[idx]
    return out


def save_palette_yaml(
    centroids: dict[str, np.ndarray],
    stats: dict[str, dict],
    output_path: Path,
    palette_version: str = "palette-v2",
) -> None:
    """Persist calibrated centroids as YAML (consumed by analyzer at startup)."""
    from cycling_photo_ai.color.palette.canonical import PALETTE_LAB

    full_palette = dict(PALETTE_LAB)  # achromatics + non-calibrated kept canonical
    for name, lab in centroids.items():
        full_palette[name] = lab

    output = {
        "palette_version": palette_version,
        "centroids": {name: [float(v) for v in lab] for name, lab in full_palette.items()},
        "calibration_stats": stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


def load_palette_yaml(path: Path) -> dict[str, np.ndarray]:
    """Load palette centroids from YAML produced by save_palette_yaml."""
    data = yaml.safe_load(path.read_text())
    return {
        name: np.array(lab, dtype=np.float64)
        for name, lab in data["centroids"].items()
    }


# ---------------------------------------------------------------------------
# CLI entry — uv run python -m cycling_photo_ai.color.palette.calibration ...
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from cycling_photo_ai.shared.paths import EXPERIMENTS_DIR

    parser = argparse.ArgumentParser(description="Calibrate palette centroids empirically")
    parser.add_argument("--run-name", default="run7_empirical_palette")
    parser.add_argument("--no-gray-world", action="store_true",
                        help="Disable Gray World during pixel collection")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output YAML path (default: experiments/color_<run-name>/palette_v2.yaml)",
    )
    args = parser.parse_args()

    crops = load_validation_set()
    print(f"Loaded {len(crops)} labeled crops\n")

    centroids, stats = calibrate_centroids(
        crops, apply_gray_world=not args.no_gray_world
    )

    print(f"{'name':12s}  {'crops':>6s}  {'pixels':>9s}  {'calibrated':>11s}")
    for name, s in stats.items():
        print(
            f"{name:12s}  {s['n_crops']:>6d}  {s['n_pixels']:>9d}  "
            f"{'✓' if s['calibrated'] else 'fallback':>11s}"
        )
    print()
    for name, s in stats.items():
        if s["calibrated"]:
            ca = s["canonical_lab"]
            em = s["empirical_lab"]
            print(
                f"  {name:10s} canonical [{ca[0]:6.1f}, {ca[1]:6.1f}, {ca[2]:6.1f}]  "
                f"empirical [{em[0]:6.1f}, {em[1]:6.1f}, {em[2]:6.1f}]"
            )

    out_path = args.out or (EXPERIMENTS_DIR / f"color_{args.run_name}" / "palette_v2.yaml")
    save_palette_yaml(centroids, stats, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
