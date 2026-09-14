import numpy as np
import streamlit as st

from cnn_core.convolution import conv2d
from viz.matrix_view import fig_matrix
from viz.widgets import editable_matrix, run_or_hint

st.title("1. Kernel & Convolution")
st.markdown(
    """
Convolution = trượt kernel qua input, mỗi vị trí nhân-từng-phần-tử rồi cộng
lại thành 1 số. Sửa kernel bên dưới để thấy output đổi theo — đây là toàn bộ
"phép màu" của 1 lớp conv, không có gì bí ẩn hơn.
"""
)

PRESETS = {
    "Tuỳ chỉnh": None,
    "Identity (giữ nguyên)": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "Box blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Edge detect (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "Sobel X (biên dọc)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y (biên ngang)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
}

DEFAULT_INPUT = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0],
        [0, 10, 50, 10, 0, 0],
        [0, 10, 10, 10, 0, 0],
        [0, 0, 0, 0, 10, 10],
        [0, 0, 0, 0, 10, 10],
    ],
    dtype=float,
)

col_input, col_kernel = st.columns(2)

with col_input:
    st.subheader("Input")
    st.caption("Sửa trực tiếp từng ô. Có 1 khối 'nhọn' ở giữa và 1 khối vuông ở góc để so sánh kernel.")
    input_matrix = editable_matrix("input_matrix", DEFAULT_INPUT)

with col_kernel:
    st.subheader("Kernel")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    if PRESETS[preset_name] is not None:
        kernel_default = PRESETS[preset_name]
    else:
        kernel_default = np.zeros((3, 3))
        kernel_default[1, 1] = 1
    kernel = editable_matrix(f"kernel_{preset_name}", kernel_default)

st.divider()

output = run_or_hint(
    conv2d,
    input_matrix,
    kernel,
    stride=1,
    padding=0,
    todo_hint="Implement `conv2d` trong `cnn_core/convolution.py` (Phase 2) để xem output ở đây.",
)

c1, c2, c3 = st.columns(3)
c1.pyplot(fig_matrix(input_matrix, title="Input", cmap="gray"))
c2.pyplot(fig_matrix(kernel, title="Kernel", cmap="RdBu"))
c3.pyplot(fig_matrix(output, title=f"Output {output.shape}", cmap="gray"))

st.caption(
    f"Input {input_matrix.shape} * kernel {kernel.shape} -> output {output.shape} "
    f"(công thức: {input_matrix.shape[0]} - {kernel.shape[0]} + 1 = {output.shape[0]})"
)
