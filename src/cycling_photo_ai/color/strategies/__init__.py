"""Color analysis strategies (ADR-018 / ADR-019).

ManualColorStrategy   — K-Means + Run 9 stack
GeminiColorStrategy   — TODO (ADR-019, S4)
"""

from __future__ import annotations

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy
from cycling_photo_ai.color.strategies.manual import ManualColorStrategy

__all__ = ["ColorAnalysisStrategy", "ManualColorStrategy", "GeminiColorStrategy"]
