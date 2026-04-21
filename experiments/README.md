# Experiments

Training outputs are saved here by each run. Directory structure:

```
experiments/
├── run1_yolo11m_baseline/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── args.yaml
│   ├── results.csv
│   └── confusion_matrix.png
├── run2_yolo11m_optimized/
│   └── ...
```

Each run is gitignored (large binaries). Configs that produced them live in `configs/training/`.

Back up weights to Google Drive after training on Colab.
