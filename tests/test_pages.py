"""Smoke test: mọi trang Streamlit render hết không có exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("path", PAGES, ids=[p.name for p in PAGES])
def test_page_renders_without_exception(path):
    at = AppTest.from_file(str(path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_training_page_flags_too_large_learning_rate():
    """lr 0.3 làm mạng kẹt ở loss ≈ ln 10 (đoán bừa) — trang phải báo lỗi rõ ràng."""
    at = AppTest.from_file(str(ROOT / "pages" / "11_Backprop_Training.py"), default_timeout=120).run()
    at.select_slider[0].set_value(0.3).run()
    assert not at.exception
    assert any("không học được" in e.value for e in at.error)

    at.select_slider[0].set_value(0.01).run()
    assert not at.error
