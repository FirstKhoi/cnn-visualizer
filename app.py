"""Trang chủ — tổng quan project + checklist trạng thái implement cnn_core/.

Chạy: streamlit run app.py
Các trang trong pages/ tự động xuất hiện ở sidebar (quy ước Streamlit).
"""

import numpy as np
import streamlit as st

st.set_page_config(page_title="CNN Visualizer", layout="wide")

st.title("CNN Visualizer")
st.markdown(
    """
App trực quan hoá CNN — bạn tự viết phần toán trong `cnn_core/`, app chỉ vẽ
lại kết quả. Xem lộ trình chi tiết từng phase trong `TODO.md`.

Mỗi trang bên sidebar tương ứng 1 khái niệm, đi theo đúng thứ tự nên học:
kernel/convolution → padding → stride → công thức shape → pooling →
activation → ghép thành block → full pipeline trên ảnh thật.
"""
)

st.divider()
st.subheader("Trạng thái implement")
st.caption("Tự động thử gọi từng hàm trong cnn_core/ — bảng này là cách nhanh nhất để biết đang ở phase nào.")


def _check(fn, *args, **kwargs) -> tuple[bool, str | None]:
    try:
        fn(*args, **kwargs)
        return True, None
    except NotImplementedError:
        return False, None
    except Exception as exc:  # implemented nhưng có bug -> vẫn tính là "đã đụng vào"
        return False, f"Lỗi khi test: {exc}"


rows = []

try:
    from cnn_core.padding import pad_matrix

    ok, err = _check(pad_matrix, np.array([[1]]), 1)
    rows.append(("Phase 1", "cnn_core/padding.py", "pad_matrix", ok, err))
except ImportError as exc:
    rows.append(("Phase 1", "cnn_core/padding.py", "pad_matrix", False, str(exc)))

try:
    from cnn_core.convolution import conv2d, conv2d_multichannel

    ok, err = _check(conv2d, np.ones((3, 3)), np.ones((2, 2)))
    rows.append(("Phase 2", "cnn_core/convolution.py", "conv2d", ok, err))

    ok, err = _check(conv2d_multichannel, np.ones((4, 4, 1)), np.ones((2, 2, 1, 1)))
    rows.append(("Phase 5", "cnn_core/convolution.py", "conv2d_multichannel", ok, err))
except ImportError as exc:
    rows.append(("Phase 2", "cnn_core/convolution.py", "conv2d", False, str(exc)))
    rows.append(("Phase 5", "cnn_core/convolution.py", "conv2d_multichannel", False, str(exc)))

try:
    from cnn_core.shapes import conv_output_shape

    ok, err = _check(conv_output_shape, 5, 3, 1, 0)
    rows.append(("Phase 3", "cnn_core/shapes.py", "conv_output_shape", ok, err))
except ImportError as exc:
    rows.append(("Phase 3", "cnn_core/shapes.py", "conv_output_shape", False, str(exc)))

try:
    from cnn_core.pooling import avg_pool2d, max_pool2d

    ok, err = _check(max_pool2d, np.ones((4, 4)), 2, 2)
    rows.append(("Phase 4", "cnn_core/pooling.py", "max_pool2d", ok, err))

    ok, err = _check(avg_pool2d, np.ones((4, 4)), 2, 2)
    rows.append(("Phase 4", "cnn_core/pooling.py", "avg_pool2d", ok, err))
except ImportError as exc:
    rows.append(("Phase 4", "cnn_core/pooling.py", "max_pool2d", False, str(exc)))
    rows.append(("Phase 4", "cnn_core/pooling.py", "avg_pool2d", False, str(exc)))

try:
    from cnn_core.activation import relu, sigmoid

    ok, err = _check(relu, np.array([1.0]))
    rows.append(("Phase 4", "cnn_core/activation.py", "relu", ok, err))

    ok, err = _check(sigmoid, np.array([1.0]))
    rows.append(("Phase 4", "cnn_core/activation.py", "sigmoid", ok, err))
except ImportError as exc:
    rows.append(("Phase 4", "cnn_core/activation.py", "relu", False, str(exc)))
    rows.append(("Phase 4", "cnn_core/activation.py", "sigmoid", False, str(exc)))

try:
    from cnn_core.block import ConvBlock

    def _try_block():
        block = ConvBlock(kernels=np.ones((2, 2, 1, 1)))
        block.forward(np.ones((4, 4, 1)))

    ok, err = _check(_try_block)
    rows.append(("Phase 5", "cnn_core/block.py", "ConvBlock", ok, err))
except ImportError as exc:
    rows.append(("Phase 5", "cnn_core/block.py", "ConvBlock", False, str(exc)))

for phase, file, name, ok, err in rows:
    status = "[DONE]" if ok else "[TODO]"
    col1, col2, col3 = st.columns([1, 3, 2])
    col1.markdown(f"`{status}` **{phase}**")
    col2.code(f"{file} :: {name}", language=None)
    col3.caption(err or ("Xong" if ok else "Chưa implement"))

done = sum(1 for row in rows if row[3])
st.progress(done / len(rows) if rows else 0)
st.caption(f"{done}/{len(rows)} hàm đã implement — chạy `pytest -v` để kiểm tra chi tiết đúng/sai.")
