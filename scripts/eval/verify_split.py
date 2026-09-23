"""Verificación independiente del split v4 y de los datasets YOLO construidos (doble chequeo, 2026-09-22).

Comprueba: 1 200 filas únicas; conteos por montón; cero duplicados exactos (SHA-256) entre montones;
cero grupos partidos; cero pares con distancia < 0,05 entre montones y cero pares consecutivos-y-parecidos
(recalculado desde los embeddings, sin usar el código del split); imágenes y etiquetas en disco = csv;
hash de 30 imágenes al azar por dataset; etiquetas de c2 y c1 = c5 remapeadas; coordenadas en [0,1];
placas por montón; proporciones de estratos.

Usage: uv run python scripts/eval/verify_split.py
"""
import csv, json, collections, hashlib, re, sys
import numpy as np
from pathlib import Path

rows = list(csv.DictReader(open("experiments/split_v4/split.csv")))
ok = True


def chk(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg); ok = ok and cond


names = [r["file_name"] for r in rows]
chk(len(rows) == 1200 and len(set(names)) == 1200, f"1200 filas únicas ({len(rows)}, únicas {len(set(names))})")
chk(set(r["split"] for r in rows) == {"train", "val", "test"}, "solo train/val/test")
print("     conteos", dict(collections.Counter(r["split"] for r in rows)))
byhash = collections.defaultdict(set)
for r in rows:
    byhash[r["sha256"]].add(r["split"])
chk(sum(len(s) > 1 for s in byhash.values()) == 0, "duplicados exactos entre montones: 0")
bygroup = collections.defaultdict(set)
for r in rows:
    bygroup[r["group_id"]].add(r["split"])
chk(all(len(s) == 1 for s in bygroup.values()), f"grupos partidos: {sum(len(s) > 1 for s in bygroup.values())} de {len(bygroup)}")
E = np.load("experiments/split_v4/embeddings.npy"); files = json.load(open("experiments/split_v4/embeddings_files.json"))
chk(set(files) == set(names), "embeddings cubren las 1200 fotos")
idx = {f: i for i, f in enumerate(files)}; split = {r["file_name"]: r["split"] for r in rows}; orig = {r["file_name"]: r["orig_name"] for r in rows}
D = 1 - E @ E.T; sp = np.array([split[f] for f in files]); cross = sp[:, None] != sp[None, :]; triu = np.triu(np.ones_like(D, bool), 1)
chk(int(((D < 0.05) & cross & triu).sum()) == 0, "pares con distancia <0,05 entre montones: 0")
adj = 0; bycam = collections.defaultdict(list)
for f in files:
    m = re.match(r"(DSC|_MG_|P|sultana_|el_calvario_)(\d+)", orig[f])
    if m:
        bycam[m.group(1)].append((int(m.group(2)), idx[f]))
for lst in bycam.values():
    lst.sort()
    for a in range(len(lst)):
        for b in range(a + 1, len(lst)):
            if lst[b][0] - lst[a][0] > 3:
                break
            i, j = lst[a][1], lst[b][1]
            if D[i, j] < 0.10 and sp[i] != sp[j]:
                adj += 1
chk(adj == 0, f"pares consecutivos y parecidos entre montones: {adj}")
print("     distancia mínima entre montones:", round(float(D[cross & triu].min()), 3))
sha = {r["file_name"]: r["sha256"] for r in rows}
for c in ["c5", "c2", "c1"]:
    if not Path(f"data/v4_{c}").exists():
        continue
    for s in ["train", "val", "test"]:
        imgs = sorted(p.name for p in Path(f"data/v4_{c}/images/{s}").iterdir())
        lbls = sorted(p.stem for p in Path(f"data/v4_{c}/labels/{s}").iterdir())
        exp = sorted(r["file_name"] for r in rows if r["split"] == s)
        chk(imgs == exp and lbls == [Path(x).stem for x in exp], f"{c}/{s}: {len(imgs)} imgs, {len(lbls)} labels = csv {len(exp)}")
    rng = np.random.default_rng(0); sample = rng.choice(rows, 30, replace=False)
    bad = [r["file_name"] for r in sample if hashlib.sha256(open(f"data/v4_{c}/images/{r['split']}/{r['file_name']}", "rb").read()).hexdigest() != r["sha256"]]
    chk(not bad, f"{c}: hash de 30 imágenes al azar coincide")
maps = {"c2": {1: 0, 3: 1}, "c1": {1: 0}}
plates = collections.Counter()
for s in ["train", "val", "test"]:
    for lp in Path(f"data/v4_c5/labels/{s}").iterdir():
        l5 = [x.split() for x in lp.read_text().splitlines() if x.strip()]
        for x in l5:
            assert 0 <= int(x[0]) < 5 and all(0 <= float(v) <= 1 for v in x[1:]), lp
            if int(x[0]) == 1:
                plates[s] += 1
        for c, m in maps.items():
            if not Path(f"data/v4_{c}").exists():
                continue
            lc = [x.split() for x in Path(f"data/v4_{c}/labels/{s}/{lp.name}").read_text().splitlines() if x.strip()]
            exp = [[str(m[int(x[0])])] + x[1:] for x in l5 if int(x[0]) in m]
            if lc != exp:
                chk(False, f"{c}/{s}/{lp.name} etiquetas no coinciden con c5 remapeado")
chk(dict(plates) == {"train": 597, "val": 130, "test": 128}, f"placas por montón desde etiquetas: {dict(plates)}")
for s in ["train", "val", "test"]:
    sel = [r for r in rows if r["split"] == s]
    print(f"     {s}: fotos con placa {sum(int(r['n_plates']) > 0 for r in sel) / len(sel):.2f}, grandes {sum(r['size_bucket'] == 'large' for r in sel) / len(sel):.2f}")
print("\nRESULTADO:", "TODO OK" if ok else "HAY FALLOS")
sys.exit(0 if ok else 1)
