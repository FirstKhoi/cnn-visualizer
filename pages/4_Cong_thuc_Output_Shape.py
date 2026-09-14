import streamlit as st

from cnn_core.shapes import conv_output_shape
from viz.widgets import run_or_hint

st.title("4. Công thức Output Shape")
st.markdown(
    """
Ráp lại 3 khái niệm đã học: **W** (kích thước input), **K** (kernel),
**S** (stride), **P** (padding).

```
output = (W - K + 2*P) / S + 1
```

Nhập số vào đây, tự tính nhẩm trước, rồi xem app có ra đúng số bạn nghĩ
không. Thử cả bộ số làm phép chia không chia hết — đó là bộ tham số **không
hợp lệ** (kernel sẽ hụt chân ở cuối input).
"""
)

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

st.success(f"Output size = {result}")

st.divider()
st.caption(
    "Thử ví dụ 'vỡ' công thức: W=6, K=3, S=2, P=0 -> (6-3)/2 = 1.5, không "
    "chia hết -> conv_output_shape phải raise ValueError thay vì trả về số sai."
)
