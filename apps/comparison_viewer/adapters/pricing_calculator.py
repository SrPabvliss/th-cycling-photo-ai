from __future__ import annotations


def calculate_cost_tokens(
    pricing: dict,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    thinking_tokens: int = 0,
) -> float:
    fresh_input = max(0, input_tokens - cached_input_tokens)
    cost = 0.0
    cost += fresh_input * pricing.get("input_per_1m_usd", 0.0) / 1_000_000
    if cached_input_tokens:
        rate = pricing.get("cached_input_per_1m_usd",
                           pricing.get("input_per_1m_usd", 0.0))
        cost += cached_input_tokens * rate / 1_000_000
    cost += output_tokens * pricing.get("output_per_1m_usd", 0.0) / 1_000_000
    if thinking_tokens:
        rate = pricing.get("thinking_per_1m_usd",
                           pricing.get("output_per_1m_usd", 0.0))
        cost += thinking_tokens * rate / 1_000_000
    return cost


def calculate_cost_per_call(pricing: dict) -> float:
    unit = pricing.get("unit")
    if unit == "per_inference":
        return float(pricing.get("paid_per_inference_usd", 0.0))
    if unit == "per_1k_units":
        return float(pricing.get("paid_per_1k_usd", 0.0)) / 1000.0
    if unit == "per_1k_images":
        return float(pricing.get("paid_per_1k_usd", 0.0)) / 1000.0
    return 0.0
