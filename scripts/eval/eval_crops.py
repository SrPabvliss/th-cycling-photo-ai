"""Q-7: evaluador único del barrido OCR (ESWA). Lee `preds_{val,test}.json` de cada corrida.

Por corrida: EM (coincidencia exacta), EM a cobertura 80 / 90 / 95 % con umbral leído en
validación (se aplica a prueba y se reporta la cobertura real), área bajo la curva
exactitud-cobertura, ECE de secuencia (15 bins, confianza = mínimo por token predicho), temperatura
ajustada en validación (NLL de los tokens de la etiqueta sobre las log-probabilidades por paso) y ECE
tras temperatura, errores por tipo (sustitución, inserción, omisión, otro), val EM media de las últimas
10 épocas de la fase 3 (para declarar el sesgo del máximo sobre épocas). Por configuración (varias
semillas): media ± sd. Entre configuraciones: bootstrap pareado **por grupo de dorsal** (`group_id` de
`split.csv`; los recortes de un mismo dorsal no son independientes), sobre la media de semillas, en val y
prueba; McNemar (misma semilla) en prueba con corrección de Holm sobre todas las comparaciones de esa semilla.

Versión 2 (2026-09-25, tras revisión externa): une cada predicción con `split.csv` por índice y **verifica la
etiqueta** (las corridas de SVTRv2 se volcaron en otro orden y hubo que reordenarlas: `reorder_svtrv2_preds.py`);
formato genérico de `logp`: filas por paso, columnas = vocabulario del lector; `label_ids` opcional
(TrOCR: posiciones en `vocab`); por defecto (PARSeq, SVTRv2) columna 0 = EOS/blank y dígito d = d + 1.

Usage:
    uv run python scripts/eval/eval_crops.py --runs experiments/sweep_ocr/runs --out experiments/sweep_ocr/tabla_ocr
Convención de nombres: <config>_s<seed>[...]; la configuración es el nombre sin el sufijo de semilla.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COVERAGES = (0.80, 0.90, 0.95)
N_BINS = 15
N_BOOT = 10_000
SEED_RE = re.compile(r"_s(\d+)")
LAST_K = 10


def load(run_dir: Path, split: str, rows: list[dict]) -> list[dict]:
    raw = json.load(open(run_dir / f"preds_{split}.json"))
    preds = raw["preds"] if isinstance(raw, dict) else raw
    cr = [r for r in rows if r["split"] == split]
    if len(cr) != len(preds):
        raise SystemExit(f"{run_dir.name} {split}: {len(preds)} predicciones y {len(cr)} filas en split.csv")
    for k, (p, r) in enumerate(zip(preds, cr)):
        if p["label"] != r["bib_number"]:
            raise SystemExit(f"{run_dir.name} {split}: etiqueta {p['label']} en posición {k} no coincide con split.csv ({r['bib_number']}, {r['crop_file']}); orden de volcado incorrecto")
        p.setdefault("crop_file", r["crop_file"])
        p["group_id"] = r["group_id"]
    return preds


def correct(p: dict) -> bool:
    return p["pred"] == p["label"]


def n_tokens(p: dict) -> int:
    return len(p.get("probs", []))


def confidence(p: dict, T: float = 1.0) -> float:
    """Mínimo sobre los tokens predichos de la probabilidad del token elegido, con temperatura T sobre logp."""
    if not p["pred"] or n_tokens(p) == 0:
        return 0.0
    if T == 1.0 or not p.get("logp"):
        return float(p["conf_min"])
    rows = np.asarray(p["logp"][: n_tokens(p)], dtype=float) / T
    rows = rows - rows.max(-1, keepdims=True)
    probs = np.exp(rows) / np.exp(rows).sum(-1, keepdims=True)
    return float(probs.max(-1).min())


def label_token_ids(p: dict) -> list[int]:
    """Columnas de `logp` que corresponden a la etiqueta. Por defecto (PARSeq, SVTRv2): EOS/blank = 0, dígito d = d + 1."""
    if "label_ids" in p:
        return list(p["label_ids"])
    return [int(d) + 1 for d in p["label"]] + [0]


def fit_temperature(preds: list[dict]) -> float:
    """T que minimiza la NLL de los tokens de la etiqueta en val sobre las filas volcadas (decodificación voraz). Malla fina."""
    rows, tgts = [], []
    for p in preds:
        if not p.get("logp"):
            continue
        ids = label_token_ids(p)
        for r, t in zip(p["logp"][: len(ids)], ids):
            rows.append(np.asarray(r, dtype=float))
            tgts.append(t)
    if not rows:
        return 1.0
    L = np.stack(rows)
    tgt = np.asarray(tgts)
    best_T, best_nll = 1.0, math.inf
    for T in np.concatenate([np.arange(0.2, 1.0, 0.02), np.arange(1.0, 5.0, 0.05)]):
        z = L / T
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        nll = -logp[np.arange(len(tgt)), tgt].mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def ece(confs: np.ndarray, corr: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - confs[m].mean())
    return float(e)


def acc_cov_auc(confs: np.ndarray, corr: np.ndarray) -> float:
    order = np.argsort(-confs)
    c = corr[order]
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    cov = np.arange(1, len(c) + 1) / len(c)
    return float(np.trapezoid(acc, cov))


def thresholds_from_val(confs_val: np.ndarray) -> dict:
    out = {}
    s = np.sort(confs_val)[::-1]
    for cv in COVERAGES:
        k = max(1, int(math.floor(cv * len(s))))
        out[cv] = float(s[k - 1])
    return out


def error_type(p: dict) -> str:
    a, b = p["label"], p["pred"]
    if a == b:
        return "ok"
    if len(a) == len(b):
        return "sustitucion" if sum(x != y for x, y in zip(a, b)) == 1 else "multiple"
    if len(b) == len(a) + 1 and any(b[:i] + b[i + 1:] == a for i in range(len(b))):
        return "insercion"
    if len(b) == len(a) - 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a))):
        return "omision"
    return "otro"


def eval_run(run_dir: Path, rows: list[dict]) -> dict:
    val, test = load(run_dir, "val", rows), load(run_dir, "test", rows)
    T = fit_temperature(val)
    res = {"run": run_dir.name, "T": T, "n_val": len(val), "n_test": len(test)}
    for split, preds in (("val", val), ("test", test)):
        corr = np.array([correct(p) for p in preds], dtype=float)
        c1 = np.array([confidence(p) for p in preds])
        cT = np.array([confidence(p, T) for p in preds])
        res[f"{split}_em"] = float(corr.mean())
        res[f"{split}_ece"] = ece(c1, corr)
        res[f"{split}_ece_T"] = ece(cT, corr)
        res[f"{split}_auc"] = acc_cov_auc(c1, corr)
        res[f"{split}_conf_mean"] = float(c1.mean())
        res[f"{split}_errors"] = dict(Counter(error_type(p) for p in preds if not correct(p)))
    thr = thresholds_from_val(np.array([confidence(p) for p in val]))
    corr_t = np.array([correct(p) for p in test], dtype=float)
    conf_t = np.array([confidence(p) for p in test])
    for cv, t in thr.items():
        m = conf_t >= t
        res[f"test_em@{int(cv*100)}"] = float(corr_t[m].mean()) if m.any() else float("nan")
        res[f"test_cov@{int(cv*100)}"] = float(m.mean())
        res[f"thr@{int(cv*100)}"] = t
    meta_p = run_dir / "run_meta.json"
    res["val_em_last"] = float("nan")
    if meta_p.exists():
        meta = json.load(open(meta_p))
        res["params_M"] = meta.get("params_M")
        res["train_seconds"] = sum(r.get("train_seconds", 0) for r in meta.get("phase_results", {}).values())
        res["best_epochs"] = {p: r.get("best_epoch") for p, r in meta.get("phase_results", {}).items()}
        last = list(meta.get("phase_results", {}).values())[-1].get("history", []) if meta.get("phase_results") else []
        hist = [h["val_em"] for h in last if h.get("epoch", 0) > 0]
        if hist:
            res["val_em_last"] = float(np.mean(hist[-LAST_K:]))
    res["_corr_test"] = corr_t.tolist()
    res["_corr_val"] = np.array([correct(p) for p in val], dtype=float).tolist()
    res["_groups_test"] = [p["group_id"] for p in test]
    res["_groups_val"] = [p["group_id"] for p in val]
    return res


def config_of(run_name: str) -> str:
    return SEED_RE.sub("", run_name, count=1)


def paired_bootstrap_groups(a: np.ndarray, b: np.ndarray, groups: list, n_boot: int, rng: np.random.Generator) -> dict:
    """Bootstrap pareado por grupo (dorsal): se remuestrean grupos con reemplazo y se concatenan sus recortes."""
    gids = sorted(set(groups))
    idx_by_g = {g: np.where(np.asarray(groups) == g)[0] for g in gids}
    d_crop = a - b
    sums = np.array([d_crop[idx_by_g[g]].sum() for g in gids])
    sizes = np.array([len(idx_by_g[g]) for g in gids])
    pick = rng.integers(0, len(gids), size=(n_boot, len(gids)))
    d = sums[pick].sum(1) / sizes[pick].sum(1)
    return {"diff": float(d_crop.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt0": float((d > 0).mean()), "n_groups": len(gids)}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return {"b": n01, "c": n10, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)  # binomial exacta, dos colas
    return {"b": n01, "c": n10, "p": float(p)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-csv", default="experiments/ocr_split_v2/split.csv")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.split_csv)))
    runs_dir = Path(args.runs)
    run_dirs = sorted(d for d in runs_dir.iterdir() if (d / "preds_test.json").exists())
    results = [eval_run(d, rows) for d in run_dirs]
    by_cfg = defaultdict(list)
    for r in results:
        by_cfg[config_of(r["run"])].append(r)

    keys = ["val_em", "val_em_last", "test_em", "test_em@80", "test_em@90", "test_em@95", "test_cov@80", "test_cov@90", "test_cov@95",
            "test_auc", "test_ece", "test_ece_T", "test_conf_mean", "T", "train_seconds"]
    table = []
    for cfg, rs in sorted(by_cfg.items(), key=lambda kv: -np.mean([r["val_em"] for r in kv[1]])):
        row = {"config": cfg, "seeds": len(rs), "params_M": rs[0].get("params_M")}
        for k in keys:
            vals = np.array([r[k] for r in rs], dtype=float)
            row[k] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            row[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["test_errors"] = dict(sum((Counter(r["test_errors"]) for r in rs), Counter()))
        table.append(row)

    rng = np.random.default_rng(args.seed)
    pairs = []
    cfgs = [t["config"] for t in table]
    groups_t, groups_v = results[0]["_groups_test"], results[0]["_groups_val"]
    for i in range(len(cfgs)):
        for j in range(i + 1, len(cfgs)):
            a = np.mean([r["_corr_test"] for r in by_cfg[cfgs[i]]], axis=0)
            b = np.mean([r["_corr_test"] for r in by_cfg[cfgs[j]]], axis=0)
            av = np.mean([r["_corr_val"] for r in by_cfg[cfgs[i]]], axis=0)
            bv = np.mean([r["_corr_val"] for r in by_cfg[cfgs[j]]], axis=0)
            seeds_i = {SEED_RE.search(r["run"]).group(1): r for r in by_cfg[cfgs[i]] if SEED_RE.search(r["run"])}
            seeds_j = {SEED_RE.search(r["run"]).group(1): r for r in by_cfg[cfgs[j]] if SEED_RE.search(r["run"])}
            mc = {s: mcnemar(np.array(seeds_i[s]["_corr_test"]), np.array(seeds_j[s]["_corr_test"]))
                  for s in sorted(set(seeds_i) & set(seeds_j))}
            pairs.append({"a": cfgs[i], "b": cfgs[j],
                          "test": paired_bootstrap_groups(a, b, groups_t, args.n_boot, rng),
                          "val": paired_bootstrap_groups(av, bv, groups_v, args.n_boot, rng),
                          "mcnemar_test_by_seed": mc})
    # Holm por semilla sobre todas las comparaciones que comparten esa semilla
    for s in sorted({s for p in pairs for s in p["mcnemar_test_by_seed"]}):
        adj = holm({k: p["mcnemar_test_by_seed"][s]["p"] for k, p in enumerate(pairs) if s in p["mcnemar_test_by_seed"]})
        for k, q in adj.items():
            pairs[k]["mcnemar_test_by_seed"][s]["p_holm"] = q

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    json.dump({"runs": slim, "table": table, "pairs": pairs, "n_boot": args.n_boot, "bootstrap": "pareado por grupo de dorsal (group_id), media de semillas",
               "mcnemar": "exacto, dos colas, por semilla; p_holm = Holm sobre todas las comparaciones de esa semilla"},
              open(out.with_suffix(".json"), "w"), indent=2)

    def f(x, sd=None, pct=True):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        v = f"{100*x:.1f}" if pct else f"{x:.2f}"
        return v + (f" ± {100*sd:.1f}" if sd is not None and pct else (f" ± {sd:.2f}" if sd is not None else ""))

    lines = ["| Configuración | Semillas | Params (M) | Val EM (mejor época) | Val EM (media últ. 10 ép.) | Prueba EM | EM@80 (cob.) | EM@90 (cob.) | EM@95 (cob.) | AUC acc-cob | ECE | ECE con T | T | Entren. (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append("| " + " | ".join([
            t["config"], str(t["seeds"]), f"{t['params_M']:.1f}" if t.get("params_M") else "—",
            f(t["val_em"], t["val_em_sd"]), f(t["val_em_last"], t["val_em_last_sd"]), f(t["test_em"], t["test_em_sd"]),
            f"{f(t['test_em@80'], t['test_em@80_sd'])} ({100*t['test_cov@80']:.0f} %)",
            f"{f(t['test_em@90'], t['test_em@90_sd'])} ({100*t['test_cov@90']:.0f} %)",
            f"{f(t['test_em@95'], t['test_em@95_sd'])} ({100*t['test_cov@95']:.0f} %)",
            f"{t['test_auc']:.3f}", f"{t['test_ece']:.3f}", f"{t['test_ece_T']:.3f}", f"{t['T']:.2f}", f"{t['train_seconds']:.0f}",
        ]) + " |")
    lines += ["", "Bootstrap pareado por grupo de dorsal (37 grupos por partición), media de semillas, 10 000 remuestreos. McNemar exacto por semilla; entre paréntesis p de Holm sobre todas las comparaciones de esa semilla.",
              "", "| Comparación | Val: dif. [IC95] | Prueba: dif. [IC95] | McNemar prueba (p por semilla, Holm) |", "|---|---|---|---|"]
    for p in pairs:
        v, t = p["val"], p["test"]
        mc = ", ".join(f"s{s}: {m['p']:.3f} ({m.get('p_holm', float('nan')):.3f})" for s, m in p["mcnemar_test_by_seed"].items()) or "—"
        lines.append(f"| {p['a']} − {p['b']} | {100*v['diff']:+.1f} [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}] | "
                     f"{100*t['diff']:+.1f} [{100*t['ci95'][0]:+.1f}, {100*t['ci95'][1]:+.1f}] | {mc} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
