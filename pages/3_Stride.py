import numpy as np
import streamlit as st

from cnn_core.convolution import conv2d
from viz.matrix_view import fig_matrix_with_highlight, fig_matrix
from viz.sliding_window import window_positions
from viz.widgets import run_or_hint

st.title("3. Stride")
st.markdown(
    """
Stride = bước nhảy của kernel mỗi lần trượt. stride=1 trượt từng pixel một
(không bỏ sót gì); stride>1 "nhảy cóc" — output nhỏ đi nhanh hơn, nhưng bỏ
qua một số vị trí input hoàn toàn không được kernel nhìn tới.
"""
)

DEFAULT_INPUT = np.arange(1, 37, dtype=float).reshape(6, 6)

kernel_size = st.select_slider("Kích thước kernel", options=[2, 3], value=2)
stride = st.select_slider("Stride", options=[1, 2, 3], value=1)
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

positions = window_positions(DEFAULT_INPUT.shape[0], kernel_size, stride)

st.divider()
st.subheader("Bước qua từng vị trí kernel trượt tới")
step = st.slider("Vị trí", min_value=0, max_value=max(len(positions) - 1, 0), value=0)
if positions:
    row, col = positions[step]
    st.pyplot(
        fig_matrix_with_highlight(
            DEFAULT_INPUT,
            highlight_top_left=(row, col),
            highlight_size=kernel_size,
            title=f"Vị trí {step + 1}/{len(positions)} — góc trên-trái ({row},{col})",
            cmap="gray",
        )
    )
else:
    st.warning("Không có vị trí hợp lệ nào với bộ tham số này.")

st.divider()
st.subheader("Output đầy đủ")
output = run_or_hint(
    conv2d,
    DEFAULT_INPUT,
    kernel,
    stride=stride,
    padding=0,
    todo_hint="Implement `conv2d` trong `cnn_core/convolution.py` (Phase 2), có hỗ trợ tham số stride.",
)
col1, col2 = st.columns(2)
col1.pyplot(fig_matrix(DEFAULT_INPUT, title=f"Input {DEFAULT_INPUT.shape}", cmap="gray"))
col2.pyplot(fig_matrix(output, title=f"Output {output.shape} (stride={stride})", cmap="gray"))

st.caption(
    f"{len(positions)} vị trí kernel trượt qua, tạo ra output {output.shape}. "
    f"stride càng lớn, càng ít vị trí được duyệt -> output càng nhỏ."
)
