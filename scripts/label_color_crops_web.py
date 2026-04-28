"""Web-based color labeling tool — dropdowns, palette swatches, mouse-friendly.

Starts a tiny FastAPI server on http://localhost:8765 with:
- crop image (resized for visibility)
- two dropdowns (top1 required, top2 optional)
- visual palette swatches as a quick-pick row
- skip / back / quit buttons
- progress bar + region tag

State is the same JSONL as label_color_crops.py
(data/color/labels/validation.jsonl), so you can switch between CLI and
web tools freely.

Usage:
    uv run python scripts/label_color_crops_web.py
    uv run python scripts/label_color_crops_web.py --port 8765
    uv run python scripts/label_color_crops_web.py --region helmet
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from cycling_photo_ai.color.palette.canonical import PALETTE_LAB, PALETTE_NAMES
from cycling_photo_ai.shared.paths import COLOR_CROPS_DIR, COLOR_LABELS_DIR

LABELS_JSONL = COLOR_LABELS_DIR / "validation.jsonl"
ACROMATIC = "acromatico"

# Palette swatches as approximate sRGB hex (referential — for UI only)
PALETTE_HEX: dict[str, str] = {
    "rojo":      "#dc1414",
    "naranja":   "#ff8c00",
    "amarillo":  "#ffdc00",
    "verde":     "#00a03c",
    "azul":      "#0a3cc8",
    "celeste":   "#87ceeb",
    "morado":    "#8232aa",
    "rosa":      "#ff96b4",
    "fucsia":    "#dc1e82",
    "marron":    "#6e3c1e",
    "negro":     "#0f0f0f",
    "gris":      "#808080",
    "blanco":    "#f5f5f5",
    "dorado":    "#d4af37",
    "plateado":  "#bebec3",
    ACROMATIC:   "#cccccc",
}


def load_metadata() -> list[dict]:
    metadata_csv = COLOR_CROPS_DIR / "metadata.csv"
    if not metadata_csv.exists():
        sys.exit(f"No metadata at {metadata_csv}. Run extract_color_crops.py first.")
    with open(metadata_csv) as f:
        return list(csv.DictReader(f))


def load_labels() -> dict[str, dict]:
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


def build_app(metadata: list[dict], region_filter: str | None) -> FastAPI:
    if region_filter:
        metadata = [r for r in metadata if r["region"] == region_filter]

    state = {
        "metadata": metadata,
        "labels": load_labels(),
        "history": [],  # crop_files in label order, for undo
    }

    app = FastAPI()

    def next_unlabeled_index() -> int | None:
        for i, row in enumerate(state["metadata"]):
            if row["crop_file"] not in state["labels"]:
                return i
        return None

    def stats() -> tuple[int, int, int]:
        total = len(state["metadata"])
        labeled = sum(
            1 for r in state["metadata"]
            if r["crop_file"] in state["labels"]
            and state["labels"][r["crop_file"]].get("top1")
        )
        skipped = sum(
            1 for r in state["metadata"]
            if r["crop_file"] in state["labels"]
            and state["labels"][r["crop_file"]].get("notes") == "skipped"
        )
        return total, labeled, skipped

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        idx = next_unlabeled_index()
        total, labeled, skipped = stats()

        if idx is None:
            return HTMLResponse(_html_done(total, labeled, skipped))

        row = state["metadata"][idx]
        return HTMLResponse(_html_form(row, idx, total, labeled, skipped))

    @app.get("/crop/{crop_file:path}")
    async def serve_crop(crop_file: str) -> Response:
        path = COLOR_CROPS_DIR / crop_file
        if not path.exists():
            return Response(status_code=404)
        return Response(content=path.read_bytes(), media_type="image/jpeg")

    @app.post("/label")
    async def submit_label(
        crop_file: str = Form(...),
        region: str = Form(...),
        action: str = Form(...),
        top1: str = Form(""),
        top2: str = Form(""),
        notes: str = Form(""),
    ) -> RedirectResponse:
        if action == "back":
            if state["history"]:
                last = state["history"].pop()
                state["labels"].pop(last, None)
                save_labels(state["labels"])
            return RedirectResponse("/", status_code=303)

        if action == "skip":
            entry = {
                "crop_file": crop_file,
                "region": region,
                "top1": None,
                "top2": None,
                "notes": "skipped",
            }
        else:  # save
            top1 = top1.strip()
            top2_clean = top2.strip() or None
            if not top1:
                return RedirectResponse("/", status_code=303)
            entry = {
                "crop_file": crop_file,
                "region": region,
                "top1": top1,
                "top2": top2_clean,
                "notes": notes.strip(),
            }

        state["labels"][crop_file] = entry
        state["history"].append(crop_file)
        save_labels(state["labels"])
        return RedirectResponse("/", status_code=303)

    return app


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _palette_options(selected: str = "") -> str:
    opts = ['<option value="">— none —</option>']
    for name in PALETTE_NAMES + [ACROMATIC]:
        sel = " selected" if name == selected else ""
        opts.append(f'<option value="{name}"{sel}>{name}</option>')
    return "\n".join(opts)


def _palette_swatch_row(target_id: str) -> str:
    """Clickable swatches that fill a target select element."""
    cells = []
    for name in PALETTE_NAMES + [ACROMATIC]:
        hex_color = PALETTE_HEX.get(name, "#888888")
        cells.append(
            f'''<button type="button"
                  class="swatch"
                  style="background:{hex_color};"
                  title="{name}"
                  onclick="document.getElementById('{target_id}').value='{name}'">
                  <span class="swatch-label">{name}</span>
                </button>'''
        )
    return "\n".join(cells)


def _html_form(row: dict, idx: int, total: int, labeled: int, skipped: int) -> str:
    crop_file = row["crop_file"]
    region = row["region"]
    progress_pct = int(100 * (labeled + skipped) / max(1, total))

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Color labeler — {idx + 1}/{total}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
          background: #1a1a1a; color: #eee; }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
  .progress {{ height: 8px; background: #333; border-radius: 4px; overflow: hidden;
               margin-bottom: 16px; }}
  .progress-bar {{ height: 100%; background: #4ade80; width: {progress_pct}%;
                    transition: width .2s; }}
  .meta {{ display: flex; justify-content: space-between; font-size: 14px;
          color: #aaa; margin-bottom: 12px; }}
  .region-tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
                 background: #2a4a8a; color: #fff; font-size: 13px; font-weight: 600; }}
  .crop-wrap {{ background: #000; border-radius: 8px; padding: 16px;
                display: flex; align-items: center; justify-content: center;
                min-height: 320px; margin-bottom: 16px; }}
  .crop-wrap img {{ max-width: 100%; max-height: 480px; border-radius: 4px; }}
  label.field {{ display: block; font-size: 13px; color: #aaa; margin: 12px 0 4px; }}
  select, input[type=text] {{
    width: 100%; padding: 10px; background: #2a2a2a; color: #fff;
    border: 1px solid #444; border-radius: 4px; font-size: 16px;
  }}
  .swatch-grid {{ display: grid; grid-template-columns: repeat(8, 1fr);
                  gap: 6px; margin-top: 6px; }}
  .swatch {{ height: 48px; border: 2px solid #333; border-radius: 6px;
             cursor: pointer; position: relative; padding: 0;
             transition: border-color .1s, transform .1s; }}
  .swatch:hover {{ border-color: #fff; transform: scale(1.05); }}
  .swatch-label {{ position: absolute; bottom: 2px; left: 0; right: 0;
                    font-size: 10px; color: #000; background: rgba(255,255,255,.85);
                    padding: 1px 0; border-radius: 0 0 4px 4px; }}
  .actions {{ display: flex; gap: 8px; margin-top: 20px; }}
  button.btn {{ flex: 1; padding: 14px; font-size: 15px; border-radius: 4px;
                border: none; cursor: pointer; font-weight: 600; color: #fff; }}
  .btn-save {{ background: #4ade80; color: #000; }}
  .btn-skip {{ background: #f59e0b; }}
  .btn-back {{ background: #6b7280; }}
  .btn-save:hover {{ background: #22c55e; }}
  .btn-skip:hover {{ background: #d97706; }}
  .btn-back:hover {{ background: #4b5563; }}
  kbd {{ background: #333; padding: 1px 6px; border-radius: 3px; font-size: 11px;
         color: #aaa; }}
</style>
</head>
<body>
<div class="container">
  <div class="progress"><div class="progress-bar"></div></div>
  <div class="meta">
    <span><span class="region-tag">{region}</span>
          &nbsp;{idx + 1} / {total} — {labeled} labeled, {skipped} skipped</span>
    <span><kbd>Enter</kbd> save · <kbd>Esc</kbd> skip · <kbd>←</kbd> back</span>
  </div>

  <div class="crop-wrap">
    <img src="/crop/{crop_file}" alt="{crop_file}">
  </div>

  <form method="post" action="/label" id="form">
    <input type="hidden" name="crop_file" value="{crop_file}">
    <input type="hidden" name="region" value="{region}">
    <input type="hidden" name="action" id="action" value="save">

    <label class="field">Top 1 (dominant color) — required</label>
    <select name="top1" id="top1" required>
      {_palette_options("")}
    </select>
    <div class="swatch-grid">
      {_palette_swatch_row("top1")}
    </div>

    <label class="field">Top 2 (secondary color) — optional</label>
    <select name="top2" id="top2">
      {_palette_options("")}
    </select>
    <div class="swatch-grid">
      {_palette_swatch_row("top2")}
    </div>

    <label class="field">Notes (optional)</label>
    <input type="text" name="notes" placeholder="e.g. metallic finish, partial occlusion…">

    <div class="actions">
      <button type="button" class="btn btn-back"
              onclick="document.getElementById('action').value='back';
                       document.getElementById('form').submit()">
        ← Back / Undo
      </button>
      <button type="button" class="btn btn-skip"
              onclick="document.getElementById('action').value='skip';
                       document.getElementById('form').submit()">
        Skip
      </button>
      <button type="submit" class="btn btn-save">Save & next</button>
    </div>
  </form>
</div>

<script>
  // Keyboard shortcuts: Esc → skip, Arrow Left → back
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') {{
      document.getElementById('action').value = 'skip';
      document.getElementById('form').submit();
    }} else if (e.key === 'ArrowLeft' && !e.target.matches('input, select, textarea')) {{
      document.getElementById('action').value = 'back';
      document.getElementById('form').submit();
    }}
  }});
  // Auto-focus top1
  document.getElementById('top1').focus();
</script>
</body>
</html>
"""


def _html_done(total: int, labeled: int, skipped: int) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Done</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #1a1a1a;
          color: #eee; text-align: center; padding: 80px 20px; }}
  h1 {{ color: #4ade80; }}
  .stats {{ font-size: 18px; margin-top: 20px; }}
  .stats span {{ display: inline-block; margin: 0 16px; }}
</style>
</head>
<body>
<h1>All crops handled</h1>
<div class="stats">
  <span>Total: <b>{total}</b></span>
  <span>Labeled: <b style="color:#4ade80">{labeled}</b></span>
  <span>Skipped: <b style="color:#f59e0b">{skipped}</b></span>
</div>
<p style="margin-top:30px;color:#aaa">
  Output: <code>{LABELS_JSONL}</code><br>
  Re-run with <code>--region helmet</code> to filter or close this tab.
</p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-based color labeling tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--region", choices=["helmet", "cyclist_clothes", "bicycle"], default=None,
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open a browser",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    if not metadata:
        sys.exit("Empty metadata. Run extract_color_crops.py first.")

    app = build_app(metadata, args.region)
    url = f"http://{args.host}:{args.port}/"

    print(f"Color labeler running at {url}")
    print(f"Output: {LABELS_JSONL}")
    print("Ctrl-C to quit (progress is saved after every label).\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
