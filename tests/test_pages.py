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


def test_training_page_flags_collapse_without_batchnorm_too():
    at = AppTest.from_file(str(ROOT / "pages" / "11_Backprop_Training.py"), default_timeout=120).run()
    at.toggle[0].set_value(False).run()
    at.select_slider[0].set_value(0.3).run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("không học được" in e for e in errors)
    assert not any("≈ ln 10" in e for e in errors)  # loss có thể > 2.30 nhiều, đừng gọi là "≈"


@pytest.mark.parametrize("use_bn,label", [(True, "Conv 1 → BN → ReLU"), (False, "Conv 1 → ReLU")])
def test_training_page_shows_weight_change_and_real_layer_order(use_bn, label):
    at = AppTest.from_file(str(ROOT / "pages" / "11_Backprop_Training.py"), default_timeout=120).run()
    at.toggle[0].set_value(use_bn).run()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "ΔW" in text
    assert label in text


# Các câu từng nói sai so với chính số liệu app đo được (xem final review)
FALSE_CLAIMS = [
    "co lại ~½",
    "He init đúng giữ được std lúc khởi tạo",
    "cách chắc chắn nhất",
    "vài nghìn bước",
    "kernel random thành bộ dò nét",
    "kernel tự biến",
    # final review trang 13–14
    "gần theo kịp",
    "chuyển tốt hơn hẳn",
    "conv thắng Dense",
    "chỉ cần 1 đường thẳng",
    "12% bề rộng",
    "vùng đậm",
]


def test_page_copy_has_no_known_false_claims():
    for path in PAGES:
        source = path.read_text()
        for claim in FALSE_CLAIMS:
            assert claim not in source, f"{path.name}: {claim!r}"


def test_training_page_weight_change_copy_follows_the_measurement():
    """lr 0.1 đẩy Conv 1 đổi >100% — trang không được nói 'gần như giống hệt' nữa."""
    at = AppTest.from_file(str(ROOT / "pages" / "11_Backprop_Training.py"), default_timeout=120).run()
    at.select_slider[0].set_value(0.1).run()
    captions = " ".join(c.value for c in at.caption)
    assert "gần như giống hệt" not in captions
    assert "na ná" not in captions


def test_generalization_page_toasts_on_duplicate_run():
    at = AppTest.from_file(str(ROOT / "pages" / "12_Generalization.py"), default_timeout=120).run()
    at.button[0].click().run()  # preset 1 đã có sẵn từ lần mở đầu
    assert any("đã có trong so sánh" in t.value for t in at.toast)
    assert len(at.session_state["g_runs"]) == 1


def test_generalization_page_keeps_controls_after_visiting_another_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.switch_page("pages/12_Generalization.py").run()
    at.selectbox(key="g_preset").set_value("2. + Dropout 0.5").run()
    at.switch_page("pages/6_Activation.py").run()  # Streamlit dọn state widget trang 12
    at.switch_page("pages/12_Generalization.py").run()
    assert not at.exception
    assert at.session_state["g_preset"] == "2. + Dropout 0.5"
    assert at.session_state["g_dropout"] == 0.5


def test_generalization_presets_set_every_control():
    source = (ROOT / "pages" / "12_Generalization.py").read_text()
    presets_src = source[source.index("OVERFIT = dict(") : source.index("def apply_preset")]
    namespace = {"MAX_TRAIN_SIZE": 1297}
    exec(presets_src, namespace)

    at = AppTest.from_file(str(ROOT / "pages" / "12_Generalization.py"), default_timeout=120).run()
    for name, values in namespace["PRESETS"].items():
        at.selectbox(key="g_preset").set_value(name).run()
        assert not at.exception
        for key, value in values.items():
            assert at.session_state[f"g_{key}"] == value, (name, key)


def test_no_raw_html_in_text_elements_that_do_not_render_it():
    """st.caption / st.metric / st.markdown (không unsafe_allow_html) in nguyên thẻ HTML."""
    import ast

    for path in PAGES:
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("caption", "metric", "markdown"):
                segment = ast.get_source_segment(source, node) or ""
                if "unsafe_allow_html=True" not in segment:
                    for tag in ("<i>", "<b>", "<br>", "<code>"):
                        assert tag not in segment, f"{path.name}:{node.lineno} dùng {tag} trong {node.func.attr}"


def test_receptive_field_caption_never_contradicts_its_measurement():
    """16 cấu hình (block × kernel × pool): câu chữ phải đi theo số đo center_concentration."""
    at = AppTest.from_file(str(ROOT / "pages" / "14_Mang_nhin_thay_gi.py"), default_timeout=180).run()
    for use_pool in (True, False):
        at.toggle[0].set_value(use_pool).run()
        for k in (3, 5):
            at.select_slider[0].set_value(k).run()
            for blocks in (1, 2, 3, 4):
                at.slider[0].set_value(blocks).run()
                assert not at.exception
                caption = next(c.value for c in at.caption if c.value.startswith("Công thức"))
                where = (use_pool, k, blocks)
                assert not ("không đều" in caption and "khá đều" in caption), where
                assert "100% diện tích" not in caption, where
                if "khá đều" in caption:
                    assert "nhiều đường đi" not in caption, where


def test_why_cnn_page_reports_shift_gap_at_both_ends_with_bn_off():
    at = AppTest.from_file(str(ROOT / "pages" / "13_Vi_sao_CNN.py"), default_timeout=180).run()
    at.toggle[0].set_value(False).run()
    assert not at.exception
    caption = next(c.value for c in at.caption if c.value.startswith("Ảnh gốc"))
    assert "Ảnh dịch 1px:" in caption and caption.count("với 50 ảnh") == 2
    assert any("Không có pool" in c.value for c in at.caption)  # baseline cho biểu đồ (b)


def test_why_cnn_page_memory_uses_readable_units():
    at = AppTest.from_file(str(ROOT / "pages" / "13_Vi_sao_CNN.py"), default_timeout=180).run()
    at.select_slider[0].set_value(8).run()
    memory = list(at.table[0].value["Bộ nhớ float32"])
    assert not any(m.startswith("0.0") for m in memory), memory


def test_saliency_section_copy_follows_selection():
    at = AppTest.from_file(str(ROOT / "pages" / "14_Mang_nhin_thay_gi.py"), default_timeout=180).run()
    occlusion = next(c.value for c in at.caption if c.value.startswith("Occlusion"))
    assert "xanh" in occlusion  # map có số âm -> colormap phân kỳ, gọi đúng tên màu
    at.selectbox[0].set_value("Overfit (300 ảnh, 30% nhãn sai)").run()
    assert not any("Đổi sang mô hình overfit" in c.value for c in at.caption)
