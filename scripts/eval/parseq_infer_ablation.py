"""P-71: ablaciones de inferencia de PARSeq sobre el MISMO checkpoint (sin reentrenar, CPU).

Para cada corrida `parseq_p1-p2-p3_s*` carga `phase3/best.pt` y evalúa val y prueba del split v2 con
`decode_ar` ∈ {False, True} y `refine_iters` ∈ {0, 1, 2}. La referencia (decode_ar=False, refine_iters=1)
debe reproducir `preds_*.json` de la corrida (salvo ±1 recorte por fp32 frente a AMP).
Salida: `experiments/sweep_ocr/parseq_infer_ablation.{json,md}`.

    uv run python scripts/eval/parseq_infer_ablation.py
"""
from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))


def main() -> None:
    import lmdb
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import parseq_sweep_core as core

    repo = Path(glob.glob(str(Path.home() / ".cache/torch/hub/baudm_parseq*"))[0])
    Tok, PARSeq = core.load_strhub(repo)
    tok = Tok(core.CHARSET)
    tf = T.Compose([T.Resize(core.IMG), T.ToTensor(), T.Normalize(0.5, 0.5)])
    data = {}
    for split in ("val", "test"):
        env = lmdb.open(str(ROOT / "experiments/ocr_split_v2/lmdb" / split), readonly=True, lock=False)
        with env.begin() as txn:
            n = int(txn.get(b"num-samples"))
            imgs = [tf(Image.open(io.BytesIO(txn.get(f"image-{i:09d}".encode()))).convert("RGB")) for i in range(1, n + 1)]
            labs = [txn.get(f"label-{i:09d}".encode()).decode() for i in range(1, n + 1)]
        data[split] = (torch.stack(imgs), labs)
    variants = [(False, 0), (False, 1), (False, 2), (True, 0), (True, 1), (True, 2)]
    out = {}
    for run in sorted((ROOT / "experiments/sweep_ocr/runs").glob("parseq_p1-p2-p3_s*")):
        if run.name.endswith("_ar") or "syn2" in run.name:
            continue
        sd = torch.load(run / "phase3" / "best.pt", map_location="cpu", weights_only=True)
        out[run.name] = {}
        for ar, ri in variants:
            m = core.build_model(PARSeq, "parseq", tok, ar, ri)
            m.load_state_dict(sd)
            m.eval()
            res = {}
            for split, (x, labs) in data.items():
                preds = []
                with torch.no_grad():
                    for s in range(0, len(labs), 50):
                        logits = m(tok, x[s:s + 50])
                        preds += ["".join(c for c in t if c.isdigit()) for t in tok.decode(logits.softmax(-1))[0]]
                res[split] = sum(p == g for p, g in zip(preds, labs)) / len(labs)
            out[run.name][f"ar={ar},refine={ri}"] = res
            print(run.name, ar, ri, res, flush=True)
    json.dump(out, open(ROOT / "experiments/sweep_ocr/parseq_infer_ablation.json", "w"), indent=2)
    runs = sorted(out)
    lines = ["| decode_ar | refine_iters | " + " | ".join(f"{r.split('_s')[-1]} val / prueba" for r in runs) + " | media val / prueba |",
             "|---|---|" + "---|" * (len(runs) + 1)]
    for ar, ri in variants:
        k = f"ar={ar},refine={ri}"
        cells = [f"{100*out[r][k]['val']:.0f} / {100*out[r][k]['test']:.0f}" for r in runs]
        mv = sum(out[r][k]["val"] for r in runs) / len(runs)
        mt = sum(out[r][k]["test"] for r in runs) / len(runs)
        lines.append(f"| {ar} | {ri} | " + " | ".join(cells) + f" | {100*mv:.1f} / {100*mt:.1f} |")
    (ROOT / "experiments/sweep_ocr/parseq_infer_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
