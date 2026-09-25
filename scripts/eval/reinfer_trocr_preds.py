"""Corrección 2026-09-25 (revisión externa, hallazgo 2): el volcado original de TrOCR cortaba el bucle de
confianza en el primer token especial, y como la etiqueta se tokeniza con `<s>` al inicio el modelo emite `<s>`
en el paso 0: `probs` quedaba en `[1.0]` (probabilidad de `<s>`, no de un dígito) y `conf_min` en 1,0 para
todos los recortes. Además guardaba `logp_topk`, que `eval_crops.py` no lee (temperatura nunca ajustada).

Este script reinfiere desde `best/` (mismo checkpoint, misma decodificación restringida a dígitos, voraz) y
vuelca el formato genérico del evaluador: `probs` = probabilidad del token elegido en cada paso de dígito
(se saltan los tokens de inicio, se corta en el de fin), `conf_min` = mínimo, `logp` = log-probabilidades
sobre el vocabulario permitido (`vocab`, ids ordenados) por paso incluido el paso de fin, y `label_ids` =
posiciones en `vocab` de la tokenización canónica de la etiqueta más el token de fin (para la temperatura).
Guarda el original como `preds_{split}.v1.json` y añade `crop_file`. Prueba en CPU.

    uv run python scripts/eval/reinfer_trocr_preds.py --runs experiments/sweep_ocr/runs --data experiments/ocr_split_v2/lmdb --split-csv experiments/ocr_split_v2/split.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def reinfer(run: Path, data: Path, rows: list[dict], device: str, batch: int = 8) -> None:
    import lmdb
    import torch
    from PIL import Image
    from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel
    from trocr_sweep_core import MAX_LEN, digit_token_ids

    processor = TrOCRProcessor.from_pretrained(str(run / "best"))
    model = VisionEncoderDecoderModel.from_pretrained(str(run / "best")).to(device).eval()
    tok = processor.tokenizer
    digit_ids, special_ids = digit_token_ids(tok)
    start_ids = {i for i in (tok.cls_token_id, tok.bos_token_id) if i is not None}
    end_ids = {i for i in (tok.sep_token_id, tok.eos_token_id, tok.pad_token_id) if i is not None}
    vocab = sorted(digit_ids | special_ids)
    pos = {t: k for k, t in enumerate(vocab)}
    allowed = torch.tensor(vocab, device=device)

    class DigitOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, allowed] = 0.0
            return scores + mask

    lp = LogitsProcessorList([DigitOnly()])
    for split in ("val", "test"):
        cr = [r for r in rows if r["split"] == split]
        env = lmdb.open(str(data / split), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            n = int(txn.get(b"num-samples").decode())
            items = [(txn.get(f"image-{i+1:09d}".encode()), txn.get(f"label-{i+1:09d}".encode()).decode()) for i in range(n)]
        assert [l for _, l in items] == [r["bib_number"] for r in cr], (run.name, split)
        preds = []
        for s in range(0, n, batch):
            chunk = items[s:s + batch]
            pv = processor(images=[Image.open(io.BytesIO(b)).convert("RGB") for b, _ in chunk], return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                out = model.generate(pv, logits_processor=lp, output_scores=True, return_dict_in_generate=True, num_beams=1,
                                     do_sample=False, max_length=MAX_LEN + 2)
            texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
            seqs = out.sequences[:, 1:]  # sin decoder_start
            for b, (t, (_, g)) in enumerate(zip(texts, chunk)):
                digits = "".join(c for c in t if c.isdigit())
                per, rows_lp, chosen = [], [], []
                for step, sc in enumerate(out.scores):
                    if step >= seqs.shape[1]:
                        break
                    tid = int(seqs[b, step])
                    if tid in start_ids:
                        continue  # `<s>` aprendido de la etiqueta: no es un dígito
                    lsm = sc[b].float().log_softmax(-1)
                    rows_lp.append([round(float(lsm[v]), 5) for v in vocab])
                    chosen.append(tid)
                    if tid in end_ids:
                        break
                    per.append(float(lsm[tid].exp()))
                lab_ids = [pos[i] for i in tok(g, add_special_tokens=False).input_ids if i in pos] + [pos[tok.sep_token_id]]
                preds.append({"crop_file": cr[s + b]["crop_file"], "label": g, "pred": digits, "raw": t, "probs": per,
                              "conf_min": min(per) if per else 0.0, "chosen_ids": chosen, "logp": rows_lp,
                              "label_ids": lab_ids, "vocab_size": len(vocab)})
        f = run / f"preds_{split}.json"
        if not (run / f"preds_{split}.v1.json").exists():
            f.rename(run / f"preds_{split}.v1.json")
        json.dump({"vocab": vocab, "preds": preds}, open(f, "w"))
        em = sum(p["pred"] == p["label"] for p in preds) / n
        old = json.load(open(run / f"preds_{split}.v1.json"))
        em_old = sum(p["pred"] == p["label"] for p in old) / n
        same = sum(p["pred"] == q["pred"] for p, q in zip(preds, old))
        print(run.name, split, f"EM {em:.3f} (v1 {em_old:.3f}); mismas predicciones {same}/{n}; conf_min media {sum(p['conf_min'] for p in preds)/n:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.split_csv)))
    for run in sorted(Path(a.runs).glob("trocr_*")):
        if (run / "best").exists():
            reinfer(run, Path(a.data), rows, a.device)


if __name__ == "__main__":
    main()
