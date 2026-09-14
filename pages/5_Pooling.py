import numpy as np
import streamlit as st

from cnn_core.pooling import avg_pool2d, max_pool2d
from viz.matrix_view import fig_matrix
from viz.widgets import editable_matrix, run_or_hint

st.title("5. Pooling")
st.markdown(
    """
Pooling tóm tắt từng vùng nhỏ thành 1 số — thu nhỏ feature map và giúp model
bớt nhạy với dịch chuyển nhỏ. Khác convolution: **không có trọng số học
được**, chỉ là 1 phép thống kê cố định (max hoặc trung bình).
"""
)

DEFAULT_INPUT = np.array(
    [
        [1, 3, 2, 4],
        [5, 9, 6, 1],
        [2, 1, 8, 3],
        [4, 2, 1, 7],
    ],
    dtype=float,
)

input_matrix = editable_matrix("pooling_input", DEFAULT_INPUT)
size = st.select_slider("size (cửa sổ pooling)", options=[2, 3], value=2)
stride = st.select_slider("stride", options=[1, 2, 3], value=2)

max_out = run_or_hint(
    max_pool2d,
    input_matrix,
    size,
    stride,
    todo_hint="Implement `max_pool2d` trong `cnn_core/pooling.py` (Phase 4) để xem kết quả ở đây.",
)
avg_out = run_or_hint(
    avg_pool2d,
    input_matrix,
    size,
    stride,
    todo_hint="Implement `avg_pool2d` trong `cnn_core/pooling.py` (Phase 4) để xem kết quả ở đây.",
)

c1, c2, c3 = st.columns(3)
c1.pyplot(fig_matrix(input_matrix, title=f"Input {input_matrix.shape}", cmap="gray"))
c2.pyplot(fig_matrix(max_out, title=f"Max pool {max_out.shape}", cmap="gray"))
c3.pyplot(fig_matrix(avg_out, title=f"Avg pool {avg_out.shape}", cmap="gray"))

st.caption(
    "Max pool giữ giá trị nổi bật nhất trong mỗi vùng (nét sắc, nhạy với "
    "outlier). Avg pool làm mượt, giảm nhiễu nhưng cũng làm mờ chi tiết."
)
