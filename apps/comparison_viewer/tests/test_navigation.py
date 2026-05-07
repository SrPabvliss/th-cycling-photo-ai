"""Smoke tests for the sidebar navigation component.

We avoid spinning up Streamlit's runtime; we just verify the module imports and
exposes ``render_sidebar``. Full UI behavior is exercised manually (see PLAN
Task 4.2 Step 2) and by ``streamlit.testing.v1.AppTest`` once a manifest
fixture is available.
"""
from __future__ import annotations

import importlib


def test_navigation_module_imports() -> None:
    mod = importlib.import_module(
        "apps.comparison_viewer.components.navigation"
    )
    assert callable(mod.render_sidebar)


def test_render_sidebar_signature() -> None:
    from inspect import signature

    from apps.comparison_viewer.components.navigation import render_sidebar

    params = list(signature(render_sidebar).parameters)
    assert params == ["manifest", "settings"]
