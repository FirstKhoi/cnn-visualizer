import numpy as np
import streamlit as st

from cnn_core.padding import pad_matrix
from viz.matrix_view import fig_matrix
from viz.widgets import editable_matrix, run_or_hint

st.title("2. Padding")
st.markdown(
    """
Padding thêm viền quanh input trước khi convolve. Hai lý do dùng padding:
(1) giữ output không co lại quá nhanh qua nhiều lớp, (2) pixel ở rìa cũng
được kernel "nhìn" đủ số lần như pixel ở giữa (không thì rìa ảnh bị bỏ quên).
"""
)

DEFAULT_INPUT = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float
)

input_matrix = editable_matrix("padding_input", DEFAULT_INPUT)
pad = st.slider("padding", min_value=0, max_value=4, value=1)

padded = run_or_hint(
    pad_matrix,
    input_matrix,
    pad,
    todo_hint="Implement `pad_matrix` trong `cnn_core/padding.py` (Phase 1) để xem kết quả ở đây.",
)

col1, col2 = st.columns(2)
col1.pyplot(fig_matrix(input_matrix, title=f"Input {input_matrix.shape}", cmap="gray"))
col2.pyplot(fig_matrix(padded, title=f"Sau khi pad={pad} -> {padded.shape}", cmap="gray"))

h, w = input_matrix.shape
st.caption(
    f"Kích thước: ({h}, {w}) -> ({h} + 2*{pad}, {w} + 2*{pad}) = "
    f"({h + 2 * pad}, {w + 2 * pad}). Viền mới toàn số 0 (zero-padding)."
)
