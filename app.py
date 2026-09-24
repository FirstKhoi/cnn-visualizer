"""Trang chủ — landing page + checklist trạng thái implement cnn_core/.

Chạy: streamlit run app.py
Các trang trong pages/ tự động xuất hiện ở sidebar (quy ước Streamlit).
"""

import numpy as np
import streamlit as st

from viz.theme import BORDER, CARD_BG, MUTED, PAGE_COLORS, PRIMARY, TEXT, hero, inject_base_css, phase_pill

st.set_page_config(page_title="CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="CNN Visualizer",
    subtitle="Tự tay viết từng phép toán của CNN — kernel, padding, stride, pooling — rồi train thật, xem BatchNorm, loss và overfit hoạt động ra sao.",
    badge="Học CNN bằng cách tự xây",
)

st.markdown(
    f"""
    <div style="color:{MUTED}; font-size:1.0rem; margin-bottom:1.4rem;">
    <b>Phần 1 (trang 1–8)</b>: phần toán forward trong <code>cnn_core/</code> do <b>bạn</b> viết bằng
    vòng lặp tay — mỗi bước implement xong 1-2 hàm là mở khoá 1 trang. Chi tiết ở <code>TODO.md</code>.<br>
    <b>Phần 2 (trang 9–12)</b>: <code>cnn_core/layers.py</code> + <code>train.py</code> (numpy vectorized,
    viết sẵn, có backward) train 1 CNN thật trên ảnh chữ số 8×8 — để thấy BatchNorm, loss,
    backprop và overfit bằng số thật.<br>
    <b>Phần 3 (trang 13–14)</b>: bản chất CNN — conv khác Dense ở đâu và hơn ở đâu, mỗi neuron nhìn
    vùng nào, và dữ liệu biến đổi ra sao qua từng lớp (<code>cnn_core/analysis.py</code>).
    </div>
    """,
    unsafe_allow_html=True,
)


def _check(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except Exception:
        return False


ready: dict[int, bool] = {}

try:
    from cnn_core.convolution import conv2d

    ready[1] = _check(conv2d, np.ones((3, 3)), np.ones((2, 2)))
except ImportError:
    ready[1] = False
ready[3] = ready[1]  # Stride dùng chung conv2d với trang 1

try:
    from cnn_core.padding import pad_matrix

    ready[2] = _check(pad_matrix, np.array([[1]]), 1)
except ImportError:
    ready[2] = False

try:
    from cnn_core.shapes import conv_output_shape

    ready[4] = _check(conv_output_shape, 5, 3, 1, 0)
except ImportError:
    ready[4] = False

try:
    from cnn_core.pooling import max_pool2d

    ready[5] = _check(max_pool2d, np.ones((4, 4)), 2, 2)
except ImportError:
    ready[5] = False

try:
    from cnn_core.activation import relu

    ready[6] = _check(relu, np.array([1.0]))
except ImportError:
    ready[6] = False

try:
    from cnn_core.block import ConvBlock

    def _try_block():
        block = ConvBlock(kernels=np.ones((2, 2, 1, 1)))
        block.forward(np.ones((4, 4, 1)))

    ready[7] = _check(_try_block)
except ImportError:
    ready[7] = False
ready[8] = ready[7]  # Full pipeline dùng chung ConvBlock với trang 7

try:
    from cnn_core.train import TrainConfig, build_model, forward

    ready[9] = _check(lambda: forward(build_model(TrainConfig()), np.zeros((1, 8, 8, 1))))
except ImportError:
    ready[9] = False
for num in (10, 11, 12):  # trang 9–12 dùng chung layers.py + train.py
    ready[num] = ready[9]

try:
    from cnn_core.analysis import receptive_field

    ready[13] = ready[9] and _check(build_model, TrainConfig(arch="mlp"))
    ready[14] = ready[9] and _check(receptive_field, [(3, 1), (2, 2)])
except ImportError:
    ready[13] = ready[14] = False

PAGES = [
    (1, "pages/1_Kernel_va_Convolution.py", "Kernel & Convolution", "Trượt kernel qua input, nhân-cộng từng vị trí."),
    (2, "pages/2_Padding.py", "Padding", "Thêm viền quanh input trước khi convolve."),
    (3, "pages/3_Stride.py", "Stride", "Bước nhảy của kernel — trượt từng bước hay nhảy cóc."),
    (4, "pages/4_Cong_thuc_Output_Shape.py", "Output Shape", "Ráp kernel + padding + stride thành 1 công thức."),
    (5, "pages/5_Pooling.py", "Pooling", "Thu nhỏ feature map: giữ giá trị lớn nhất hay lấy trung bình."),
    (6, "pages/6_Activation.py", "Activation", "ReLU cắt số âm — lý do CNN học được pattern phi tuyến."),
    (7, "pages/7_Conv_Block.py", "Conv Block", "Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh."),
    (8, "pages/8_Full_Pipeline.py", "Full Pipeline", "Xếp nhiều block, forward 1 ảnh thật qua toàn bộ chuỗi."),
    (9, "pages/9_BatchNorm.py", "BatchNorm", "Chuẩn hoá từng kênh để tín hiệu không tắt/nổ qua độ sâu."),
    (10, "pages/10_Softmax_Cross_Entropy.py", "Softmax + CE", "Logits → xác suất → loss: mạng sai đến mức nào."),
    (11, "pages/11_Backprop_Training.py", "Backprop & Training", "Gradient chảy ngược, đo xem mỗi lớp thật sự học được bao nhiêu."),
    (12, "pages/12_Generalization.py", "Generalization", "Vì sao dropout, weight decay, augmentation kéo val lên."),
    (13, "pages/13_Vi_sao_CNN.py", "Vì sao là CNN?", "Locality, weight sharing, equivariance — so thẳng với MLP."),
    (14, "pages/14_Mang_nhin_thay_gi.py", "Mạng nhìn thấy gì?", "Receptive field, dữ liệu qua từng lớp, pixel nào quyết định."),
]

st.markdown("### Hành trình 14 bước")

cols = st.columns(4)
for idx, (num, path, title, desc) in enumerate(PAGES):
    color = PAGE_COLORS[num]
    done = ready.get(num, False)
    status_html = phase_pill("Xong" if done else "Chưa làm", color if done else MUTED)
    with cols[idx % 4]:
        st.markdown(
            f"""
            <div style="
                background:{CARD_BG};
                border:1.5px solid {color if done else BORDER};
                border-radius:14px;
                padding:1rem 1.1rem;
                margin-bottom:1rem;
                min-height:148px;
                box-shadow: 0 4px 14px -8px {color}55;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:{color}; font-weight:800; font-size:1.4rem;">{num}</span>
                    {status_html}
                </div>
                <div style="font-weight:700; color:{TEXT}; margin-top:0.3rem;">{title}</div>
                <div style="color:{MUTED}; font-size:0.85rem; margin-top:0.2rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(path, label="Mở trang", icon=":material/arrow_forward:")

st.divider()

done_count = sum(1 for v in ready.values() if v)
total = len(ready)
st.markdown(f"**Tiến độ tổng: {done_count}/{total} nhóm hàm đã implement**")
st.progress(done_count / total if total else 0)
st.caption("Chạy `pytest -v` để xem chi tiết test nào pass/fail trong từng file.")
