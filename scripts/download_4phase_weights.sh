#!/bin/bash
# Download 4-phase trained weights from Modal volume
# Run after all phases complete

set -e

echo "=== Downloading phase 4 weights ==="
mkdir -p weights/trocr_bib_4phase
modal volume get cycling-photo-ai-vol experiments/ocr_trocr_4phase/phase4/best/ weights/trocr_bib_4phase/

echo ""
echo "=== Downloading training summaries ==="
mkdir -p experiments/ocr_trocr_4phase
for phase in 1 2 4; do
    echo "  Phase $phase summary..."
    modal volume get cycling-photo-ai-vol \
        experiments/ocr_trocr_4phase/phase${phase}/summary.json \
        experiments/ocr_trocr_4phase/phase${phase}_summary.json 2>/dev/null || echo "  (not found)"
done

echo ""
echo "=== Done ==="
echo "Weights at: weights/trocr_bib_4phase/"
echo ""
echo "To use: set TROCR_WEIGHTS=weights/trocr_bib_4phase"
echo "Or: cp -r weights/trocr_bib_4phase/* weights/trocr_bib/"
