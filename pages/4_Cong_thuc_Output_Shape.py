import streamlit as st

from cnn_core.shapes import conv_output_shape
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import run_or_hint

COLOR = PAGE_COLORS[4]

st.set_page_config(page_title="Output Shape — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="4. Công thức Output Shape",
    subtitle="Ráp lại 3 khái niệm đã học: kernel, stride, padding — thành đúng 1 công thức.",
    badge="Phase 3",
    color=COLOR,
)
callout(
    "Không cần học thuộc — công thức này chỉ đơn giản đếm xem kernel trượt "
    "được bao nhiêu lần trên input đã pad, với mỗi bước nhảy dài `stride`. "
    "Hiểu trang 1-3 rồi thì công thức này chỉ là viết lại bằng ký hiệu.",
    color=COLOR,
    label="Vì sao quan trọng",
)

with st.container(border=True):
    st.latex(r"\text{output} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1")

    col1, col2, col3, col4 = st.columns(4)
    w = col1.number_input("W (input size)", min_value=1, value=32)
    k = col2.number_input("K (kernel size)", min_value=1, value=3)
    s = col3.number_input("S (stride)", min_value=1, value=1)
    p = col4.number_input("P (padding)", min_value=0, value=0)

    st.latex(rf"\text{{output}} = \frac{{{w} - {k} + 2 \times {p}}}{{{s}}} + 1")

    result = run_or_hint(
        conv_output_shape,
        int(w),
        int(k),
        int(s),
        int(p),
        todo_hint="Implement `conv_output_shape` trong `cnn_core/shapes.py` (Phase 3) để xem kết quả ở đây.",
    )
    st.metric("Output size", result)

st.divider()
with st.container(border=True):
    st.markdown("**Thử ví dụ 'vỡ' công thức**")
    st.caption(
        "W=6, K=3, S=2, P=0 -> (6-3)/2 = 1.5, không chia hết -> "
        "conv_output_shape phải raise ValueError thay vì trả về số sai. "
        "Nhập bộ số này ở trên để xem thông báo lỗi."
    )
