import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.activation import relu, sigmoid
from viz.matrix_view import DIV, fig_matrix
from viz.theme import ACCENT, BORDER, PAGE_COLORS, PRIMARY, TEXT, callout, hero, inject_base_css
from viz.widgets import run_or_hint

COLOR = PAGE_COLORS[6]

st.set_page_config(page_title="Activation — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="6. Activation",
    subtitle="ReLU cắt hết số âm — lý do đơn giản nhưng sống còn để CNN học được pattern phi tuyến.",
    badge="Phase 4",
    color=COLOR,
)
callout(
    "Không có non-linearity, xếp chồng nhiều lớp conv (đều là phép tuyến "
    "tính) tương đương đúng 1 lớp conv duy nhất — activation là thứ cho phép "
    "mạng học được pattern phức tạp hơn tuyến tính.",
    color=COLOR,
    label="Vì sao quan trọng",
)

DEFAULT_INPUT = np.array(
    [
        [-3, 1, -2, 4],
        [5, -1, 0, -4],
        [2, -5, 3, 1],
        [-2, 0, -1, 6],
    ],
    dtype=float,
)

with st.container(border=True):
    st.markdown("#### Trên feature map (vd: output của 1 lớp conv, có số âm lẫn dương)")
    relu_out = run_or_hint(
        relu, DEFAULT_INPUT, todo_hint="Implement `relu` trong `cnn_core/activation.py` (Phase 4)."
    )
    c1, c2 = st.columns(2)
    c1.pyplot(fig_matrix(DEFAULT_INPUT, title="Trước ReLU", cmap=DIV))
    c2.pyplot(fig_matrix(relu_out, title="Sau ReLU"))
    st.caption("Mọi giá trị âm bị cắt về 0 — feature map sau ReLU không còn số âm nào.")

with st.container(border=True):
    st.markdown("#### So sánh ReLU vs Sigmoid trên 1 đường cong")
    x = np.linspace(-6, 6, 200)
    relu_curve = run_or_hint(relu, x, todo_hint="Implement `relu` trong `cnn_core/activation.py` (Phase 4).")
    sigmoid_curve = run_or_hint(
        sigmoid, x, todo_hint="Implement `sigmoid` trong `cnn_core/activation.py` (Phase 4)."
    )

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, relu_curve, label="ReLU", color=PRIMARY, linewidth=2.5)
    ax.plot(x, sigmoid_curve, label="Sigmoid", color=ACCENT, linewidth=2.5)
    ax.axhline(0, color=BORDER, linewidth=1)
    ax.axvline(0, color=BORDER, linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, labelcolor=TEXT)
    ax.set_title("ReLU vs Sigmoid", color=TEXT, fontweight="bold")
    st.pyplot(fig)

    st.caption(
        "CNN hiện đại hầu như luôn dùng ReLU sau conv (rẻ tính, không bị "
        "'vanishing gradient' như sigmoid ở 2 đầu đường cong)."
    )
