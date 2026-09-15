import numpy as np
import streamlit as st

from cnn_core.convolution import conv2d
from viz.matrix_view import DIV, fig_matrix, fig_matrix_with_highlight
from viz.sliding_window import window_positions
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import editable_matrix, run_or_hint, step_controls

COLOR = PAGE_COLORS[1]

st.set_page_config(page_title="Kernel & Convolution — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="1. Kernel & Convolution",
    subtitle="Trượt kernel qua input, nhân-cộng từng vị trí — đó là toàn bộ 'phép màu' của convolution.",
    badge="Phase 2",
    color=COLOR,
)
callout(
    "Convolution không có gì bí ẩn: mỗi kernel là 1 'bộ dò' 1 loại pattern "
    "(cạnh, góc, vệt sáng...). Nhân kernel với từng vùng nhỏ của input rồi cộng "
    "lại cho ra 1 số đo 'vùng đó giống pattern của kernel đến mức nào'.",
    color=COLOR,
    label="Vì sao quan trọng",
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

with st.container(border=True):
    st.markdown("#### Input & Kernel")
    col_input, col_kernel = st.columns(2)
    with col_input:
        st.caption("Sửa trực tiếp từng ô. Có 1 khối 'nhọn' ở giữa và 1 khối vuông ở góc để so sánh kernel.")
        input_matrix = editable_matrix("input_matrix", DEFAULT_INPUT)
    with col_kernel:
        preset_name = st.selectbox("Preset kernel", list(PRESETS.keys()))
        if PRESETS[preset_name] is not None:
            kernel_default = PRESETS[preset_name]
        else:
            kernel_default = np.zeros((3, 3))
            kernel_default[1, 1] = 1
        kernel = editable_matrix(f"kernel_{preset_name}", kernel_default)

output = run_or_hint(
    conv2d,
    input_matrix,
    kernel,
    stride=1,
    padding=0,
    todo_hint="Implement `conv2d` trong `cnn_core/convolution.py` (Phase 2) để xem kernel trượt ở đây.",
)

st.markdown("#### Xem kernel trượt qua từng vị trí")
k = kernel.shape[0]
positions = window_positions(input_matrix.shape[0], k, stride=1)

with st.container(border=True):
    step = step_controls("kernel_walk", len(positions), color=COLOR)
    row, col = positions[step]
    window = input_matrix[row : row + k, col : col + k]
    value = float(np.sum(window * kernel))

    mask = np.zeros_like(output, dtype=bool)
    for r, c in positions[: step + 1]:
        mask[r, c] = True

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            fig_matrix_with_highlight(
                input_matrix,
                highlight_top_left=(row, col),
                highlight_size=k,
                title=f"Input — cửa sổ tại ({row}, {col})",
                highlight_color=COLOR,
            )
        )
    with c2:
        st.pyplot(fig_matrix(output, title=f"Output (đang lấp dần) {output.shape}", mask=mask))

    terms = " + ".join(f"{a:.2g}×{b:.2g}" for a, b in zip(window.flatten(), kernel.flatten()))
    st.markdown("Phép tính cho ô đang highlight:")
    st.code(f"{terms} = {value:.3g}", language=None)

st.divider()
st.markdown("#### Kernel đang dùng")
st.pyplot(fig_matrix(kernel, title="Kernel", cmap=DIV))
st.caption(
    f"Input {input_matrix.shape} * kernel {kernel.shape} -> output {output.shape} "
    f"(công thức: {input_matrix.shape[0]} - {kernel.shape[0]} + 1 = {output.shape[0]})"
)
