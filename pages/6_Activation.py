import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.activation import relu, sigmoid
from viz.matrix_view import fig_matrix
from viz.widgets import run_or_hint

st.title("6. Activation")
st.markdown(
    """
Không có non-linearity, xếp chồng nhiều lớp conv (đều là phép tuyến tính)
tương đương đúng 1 lớp conv duy nhất — activation là thứ cho phép mạng học
được pattern phức tạp hơn tuyến tính.
"""
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

st.subheader("Trên feature map (vd: output của 1 lớp conv, có số âm lẫn dương)")
relu_out = run_or_hint(
    relu, DEFAULT_INPUT, todo_hint="Implement `relu` trong `cnn_core/activation.py` (Phase 4)."
)
c1, c2 = st.columns(2)
c1.pyplot(fig_matrix(DEFAULT_INPUT, title="Trước ReLU", cmap="RdBu"))
c2.pyplot(fig_matrix(relu_out, title="Sau ReLU", cmap="gray"))
st.caption("Mọi giá trị âm bị cắt về 0 — feature map sau ReLU không còn số âm nào.")

st.divider()
st.subheader("So sánh ReLU vs Sigmoid trên 1 đường cong")
x = np.linspace(-6, 6, 200)
relu_curve = run_or_hint(relu, x, todo_hint="Implement `relu` trong `cnn_core/activation.py` (Phase 4).")
sigmoid_curve = run_or_hint(
    sigmoid, x, todo_hint="Implement `sigmoid` trong `cnn_core/activation.py` (Phase 4)."
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, relu_curve, label="ReLU")
ax.plot(x, sigmoid_curve, label="Sigmoid")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.legend()
ax.set_title("ReLU vs Sigmoid")
st.pyplot(fig)

st.caption(
    "CNN hiện đại hầu như luôn dùng ReLU sau conv (rẻ tính, không bị "
    "'vanishing gradient' như sigmoid ở 2 đầu đường cong)."
)
