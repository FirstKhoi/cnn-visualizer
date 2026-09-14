import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from cnn_core.block import ConvBlock, make_random_kernels
from viz.matrix_view import fig_feature_maps
from viz.widgets import run_or_hint

st.title("8. Full Pipeline")
st.markdown(
    """
Xếp nhiều `ConvBlock` liên tiếp và forward 1 ảnh thật qua toàn bộ chuỗi —
đúng cách LeNet/VGG hoạt động (chỉ khác: kernel ở đây random, không train).
Theo dõi shape thu nhỏ dần qua từng block, khớp với công thức đã học ở
trang 4.
"""
)


def make_synthetic_image(size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 3))
    img[: size // 2, : size // 2, 0] = 1.0
    img[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)
    rng = np.random.default_rng(1)
    img[:, :, 2] = rng.uniform(0, 0.3, size=(size, size))
    return img


uploaded = st.file_uploader("Ảnh của bạn (tuỳ chọn — không upload thì dùng ảnh tổng hợp)", type=["png", "jpg", "jpeg"])
image_size = st.select_slider("Resize ảnh về (vuông)", options=[32, 64], value=32)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB").resize((image_size, image_size))
    image = np.asarray(pil_image, dtype=float) / 255.0
else:
    image = make_synthetic_image(image_size)
    st.caption("Đang dùng ảnh tổng hợp mặc định — upload ảnh riêng ở trên để thử với dữ liệu thật.")

st.subheader("Ảnh input")
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(image)
ax.axis("off")
st.pyplot(fig)

st.divider()
st.subheader("Cấu hình pipeline")
num_blocks = st.slider("Số block", min_value=1, max_value=3, value=2)
kernels_per_block = st.slider("Số kernel mỗi block", min_value=2, max_value=8, value=4)

blocks = []
in_channels = 3
for i in range(num_blocks):
    kernels = make_random_kernels(kernels_per_block, kernel_size=3, in_channels=in_channels, seed=i)
    block = run_or_hint(
        ConvBlock,
        kernels,
        None,
        1,
        0,
        2,
        2,
        todo_hint="Implement `ConvBlock` trong `cnn_core/block.py` (Phase 5) trước khi xem pipeline.",
    )
    blocks.append(block)
    in_channels = kernels_per_block

st.divider()
st.subheader("Forward pass qua từng block")

x = image
shape_rows = [("input", x.shape)]
for i, block in enumerate(blocks):
    x = run_or_hint(
        block.forward,
        x,
        todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5).",
    )
    shape_rows.append((f"block {i + 1}", x.shape))
    st.markdown(f"**Block {i + 1}** -> shape `{x.shape}`")
    st.pyplot(fig_feature_maps(x, titles=[f"k{c}" for c in range(x.shape[-1])], max_cols=4))

st.divider()
st.subheader("Tóm tắt shape qua pipeline")
st.table({"Lớp": [r[0] for r in shape_rows], "Shape": [str(r[1]) for r in shape_rows]})
