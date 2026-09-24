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


def test_training_page_does_not_blame_small_learning_rate():
    """lr nhỏ + ít epoch cũng chưa học được, nhưng lý do là chậm, không phải 'lr quá lớn'."""
    at = AppTest.from_file(str(ROOT / "pages" / "11_Backprop_Training.py"), default_timeout=120).run()
    at.slider[0].set_value(1).run()
    at.toggle[0].set_value(False).run()
    at.select_slider[0].set_value(0.001).run()
    assert not at.exception
    assert not at.error
    assert any("lr = 0.001 nhỏ" in w.value for w in at.warning)


def test_generalization_page_preset_run_and_clear():
    at = AppTest.from_file(str(ROOT / "pages" / "12_Generalization.py"), default_timeout=120).run()
    at.selectbox(key="g_preset").set_value("2. + Dropout 0.5").run()
    assert at.session_state["g_dropout"] == 0.5
    at.button[0].click().run()
    at.button[0].click().run()  # bấm lại cùng config -> không thêm trùng
    assert len(at.session_state["g_runs"]) == 2
    assert not at.exception
    at.button[1].click().run()
    assert at.info and not at.exception
