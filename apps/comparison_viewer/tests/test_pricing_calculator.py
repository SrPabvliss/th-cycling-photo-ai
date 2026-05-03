from apps.comparison_viewer.adapters.pricing_calculator import (
    calculate_cost_tokens,
    calculate_cost_per_call,
)


def test_token_pricing_basic():
    pricing = {
        "unit": "tokens",
        "input_per_1m_usd": 1.00,
        "output_per_1m_usd": 5.00,
    }
    cost = calculate_cost_tokens(
        pricing, input_tokens=1000, output_tokens=200,
    )
    expected = (1000 * 1.00 + 200 * 5.00) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_token_pricing_with_cache_hit():
    pricing = {
        "unit": "tokens",
        "input_per_1m_usd": 1.00,
        "cached_input_per_1m_usd": 0.10,
        "output_per_1m_usd": 5.00,
    }
    cost = calculate_cost_tokens(
        pricing, input_tokens=1000, output_tokens=100,
        cached_input_tokens=900,
    )
    expected = (100 * 1.00 + 900 * 0.10 + 100 * 5.00) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_token_pricing_with_thinking():
    pricing = {
        "unit": "tokens",
        "input_per_1m_usd": 1.00,
        "output_per_1m_usd": 5.00,
        "thinking_per_1m_usd": 5.00,
    }
    cost = calculate_cost_tokens(
        pricing, input_tokens=100, output_tokens=50, thinking_tokens=500,
    )
    expected = (100 * 1.00 + 50 * 5.00 + 500 * 5.00) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_per_call_pricing():
    pricing = {"unit": "per_inference", "paid_per_inference_usd": 0.001}
    assert calculate_cost_per_call(pricing) == 0.001


def test_per_1k_pricing():
    pricing = {"unit": "per_1k_units", "paid_per_1k_usd": 1.50}
    assert abs(calculate_cost_per_call(pricing) - 0.0015) < 1e-9
