import numpy as np
import streamlit as st

from cnn_core.pooling import avg_pool2d, max_pool2d
from viz.matrix_view import fig_matrix
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import editable_matrix, run_or_hint

COLOR = PAGE_COLORS[5]

st.set_page_config(page_title="Pooling — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="5. Pooling",
    subtitle="Tóm tắt từng vùng nhỏ thành 1 số — thu nhỏ feature map, không có gì cần học.",
    badge="Phase 4",
    color=COLOR,
)
callout(
    "Khác convolution: pooling không có trọng số học được, chỉ là 1 phép "
    "thống kê cố định (max hoặc trung bình). Nó giúp model bớt nhạy với dịch "
    "chuyển nhỏ — vật thể lệch vài pixel vẫn cho ra feature map gần giống nhau.",
    color=COLOR,
    label="Vì sao quan trọng",
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

with st.container(border=True):
    input_matrix = editable_matrix("pooling_input", DEFAULT_INPUT)
    col_a, col_b = st.columns(2)
    size = col_a.select_slider("size (cửa sổ pooling)", options=[2, 3], value=2)
    stride = col_b.select_slider("stride", options=[1, 2, 3], value=2)

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

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.pyplot(fig_matrix(input_matrix, title=f"Input {input_matrix.shape}"))
    c2.pyplot(fig_matrix(max_out, title=f"Max pool {max_out.shape}"))
    c3.pyplot(fig_matrix(avg_out, title=f"Avg pool {avg_out.shape}"))

    st.caption(
        "Max pool giữ giá trị nổi bật nhất trong mỗi vùng (nét sắc, nhạy với "
        "outlier). Avg pool làm mượt, giảm nhiễu nhưng cũng làm mờ chi tiết."
    )
