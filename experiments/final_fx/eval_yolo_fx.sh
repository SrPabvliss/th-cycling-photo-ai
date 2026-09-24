#!/bin/zsh
# Evalúa los YOLO de la final (épocas fijas) con el evaluador único: val fija el umbral (máx F1), test lo aplica.
export MODAL_PROFILE=eswa-eval-4
for r in yolo11m_c1_640_s42_fx yolo11m_c1_640_s123_fx yolo11m_c1_640_s7_fx yolo11m_c1_1024_s42_fx yolo11m_c1_1024_s123_fx yolo11m_c1_1024_s7_fx yolo26m_c1_1024_s42_fx yolo26m_c1_1024_s123_fx yolo26m_c1_1024_s7_fx; do
  d=experiments/final_fx/runs/$r; sz=$(echo $r | cut -d_ -f3)
  [ -f $d/best.pt ] || uv run modal volume get --force cycling-photo-ai-sweep-vol runs/$r/weights/best.pt $d/ >/dev/null 2>&1
  [ -f $d/boxes_val/metrics.json ] || uv run python scripts/eval/eval_boxes.py --weights $d/best.pt --data data/v4_c1/data.yaml --split val --imgsz $sz --out $d/boxes_val --device mps >/dev/null 2>&1
  thr=$(uv run python -c "import json;print(json.load(open('$d/boxes_val/metrics.json'))['thr_used'])")
  [ -f $d/boxes_test/metrics.json ] || uv run python scripts/eval/eval_boxes.py --weights $d/best.pt --data data/v4_c1/data.yaml --split test --imgsz $sz --thr $thr --out $d/boxes_test --device mps >/dev/null 2>&1
  echo "evaluated $r $(uv run python -c "import json;v=json.load(open('$d/boxes_val/metrics.json'));t=json.load(open('$d/boxes_test/metrics.json'));print('val',round(v['AP50_95'],4),'test',round(t['AP50_95'],4))")"
done
