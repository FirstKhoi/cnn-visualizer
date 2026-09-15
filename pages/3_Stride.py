import numpy as np
import streamlit as st

from cnn_core.convolution import conv2d
from viz.matrix_view import fig_matrix, fig_matrix_with_highlight
from viz.sliding_window import window_positions
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import run_or_hint, step_controls

COLOR = PAGE_COLORS[3]

st.set_page_config(page_title="Stride — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="3. Stride",
    subtitle="Bước nhảy của kernel mỗi lần trượt — trượt từng pixel, hay nhảy cóc bỏ qua vài pixel?",
    badge="Phase 2",
    color=COLOR,
)
callout(
    "stride=1 trượt từng pixel một, không bỏ sót gì. stride>1 'nhảy cóc' — "
    "output nhỏ đi nhanh hơn, nhưng bỏ qua một số vị trí input hoàn toàn "
    "không được kernel nhìn tới. Đây là cách rẻ tiền nhất để giảm kích thước "
    "feature map mà không cần pooling.",
    color=COLOR,
    label="Vì sao quan trọng",
)

DEFAULT_INPUT = np.arange(1, 37, dtype=float).reshape(6, 6)

with st.container(border=True):
    col_a, col_b = st.columns(2)
    kernel_size = col_a.select_slider("Kích thước kernel", options=[2, 3], value=2)
    stride = col_b.select_slider("Stride", options=[1, 2, 3], value=1)

kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
positions = window_positions(DEFAULT_INPUT.shape[0], kernel_size, stride)

output = run_or_hint(
    conv2d,
    DEFAULT_INPUT,
    kernel,
    stride=stride,
    padding=0,
    todo_hint="Implement `conv2d` trong `cnn_core/convolution.py` (Phase 2), có hỗ trợ tham số stride.",
)

st.markdown("#### Bước qua từng vị trí kernel trượt tới")
with st.container(border=True):
    if positions:
        step = step_controls("stride_walk", len(positions), color=COLOR)
        row, col = positions[step]
        window = DEFAULT_INPUT[row : row + kernel_size, col : col + kernel_size]
        value = float(np.mean(window))

        out_i, out_j = row // stride, col // stride
        mask = np.zeros_like(output, dtype=bool)
        for idx in range(step + 1):
            r, c = positions[idx]
            mask[r // stride, c // stride] = True

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(
                fig_matrix_with_highlight(
                    DEFAULT_INPUT,
                    highlight_top_left=(row, col),
                    highlight_size=kernel_size,
                    title=f"Vị trí {step + 1}/{len(positions)} — góc trên-trái ({row},{col})",
                    highlight_color=COLOR,
                )
            )
        with c2:
            st.pyplot(fig_matrix(output, title=f"Output (đang lấp dần) {output.shape}", mask=mask))

        st.code(f"trung bình cửa sổ = {value:.3g}  ->  output[{out_i}, {out_j}]", language=None)
    else:
        st.warning("Không có vị trí hợp lệ nào với bộ tham số này.")

st.divider()
st.caption(
    f"{len(positions)} vị trí kernel trượt qua trên input {DEFAULT_INPUT.shape}, tạo ra output {output.shape}. "
    f"stride càng lớn, càng ít vị trí được duyệt -> output càng nhỏ."
)
