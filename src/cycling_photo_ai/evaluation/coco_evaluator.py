"""COCO evaluation — Level A detection metrics.

Wraps pycocotools for standardized COCO evaluation.
Single evaluator instance for all models (methodological requirement).
"""

from __future__ import annotations

from pathlib import Path

from cycling_photo_ai.evaluation.ports import EvaluationResult


class CocoEvaluator:
    """Evaluator using pycocotools.COCOeval for detection metrics."""

    def __init__(self, gt_json: str | Path, class_names: list[str] | None = None) -> None:
        self._gt_json = Path(gt_json)
        self._class_names = class_names or []

    def evaluate(self, predictions_json: str | Path, model_name: str = "") -> EvaluationResult:
        """Run COCO evaluation on prediction results.

        Args:
            predictions_json: Path to COCO-format predictions JSON.
            model_name: Identifier for this evaluation run.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(str(self._gt_json))
        coco_dt = coco_gt.loadRes(str(predictions_json))

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "mAP@0.5:0.95": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],
            "AR@10": coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
        }

        # Per-class AP
        per_class: dict[str, dict[str, float]] = {}
        if self._class_names:
            cat_ids = coco_gt.getCatIds()
            for cat_id, name in zip(cat_ids, self._class_names):
                coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval_cls.params.catIds = [cat_id]
                coco_eval_cls.evaluate()
                coco_eval_cls.accumulate()
                coco_eval_cls.summarize()
                per_class[name] = {
                    "AP@0.5:0.95": coco_eval_cls.stats[0],
                    "AP@0.5": coco_eval_cls.stats[1],
                }

        return EvaluationResult(
            model_name=model_name,
            level="A",
            metrics=metrics,
            per_class=per_class,
        )
