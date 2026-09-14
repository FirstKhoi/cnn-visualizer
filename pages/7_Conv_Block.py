import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.block import ConvBlock, make_random_kernels
from viz.matrix_view import fig_feature_maps
from viz.widgets import run_or_hint

st.title("7. Conv Block")
st.markdown(
    """
Một "block" CNN cổ điển gần như luôn là **Conv → ReLU → Pool** lặp lại.
Không train ở đây (không có backprop) — kernel sinh ngẫu nhiên chỉ để quan
sát forward pass đi qua nhiều kênh trông ra sao, giống nhìn model lúc mới
khởi tạo, trước khi train.
"""
)


def make_synthetic_rgb(size: int = 16) -> np.ndarray:
    """Ảnh RGB tổng hợp: 1 hình vuông đỏ góc trên-trái, gradient xanh lá theo
    hàng, nhiễu xanh dương nhẹ — đủ cấu trúc để thấy kernel phản ứng khác nhau."""
    img = np.zeros((size, size, 3))
    img[: size // 2, : size // 2, 0] = 1.0  # đỏ
    img[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)  # xanh lá theo hàng
    rng = np.random.default_rng(0)
    img[:, :, 2] = rng.uniform(0, 0.3, size=(size, size))  # nhiễu xanh dương
    return img


image = make_synthetic_rgb()

col_a, col_b, col_c = st.columns(3)
num_kernels = col_a.slider("Số kernel", min_value=2, max_value=8, value=4)
kernel_size = col_b.select_slider("Kernel size", options=[3, 5], value=3)
seed = col_c.number_input("Seed", min_value=0, value=0)

st.subheader("Ảnh input (tổng hợp, 3 kênh)")
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(image)
ax.axis("off")
st.pyplot(fig)

kernels = make_random_kernels(num_kernels, kernel_size, in_channels=3, seed=int(seed))

block = run_or_hint(
    ConvBlock,
    kernels,
    None,
    1,
    0,
    2,
    2,
    todo_hint="Implement `ConvBlock.__init__` trong `cnn_core/block.py` (Phase 5).",
)
output = run_or_hint(
    block.forward,
    image,
    todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5) — cần conv2d_multichannel, relu, max_pool2d đã xong trước đó.",
)

st.subheader(f"Feature maps sau Conv -> ReLU -> Pool: {output.shape}")
st.pyplot(fig_feature_maps(output, titles=[f"kernel {i}" for i in range(num_kernels)]))

st.caption(
    f"Input {image.shape} -> {num_kernels} kernel {kernel_size}x{kernel_size} -> "
    f"output {output.shape}. Mỗi ô là 1 feature map — kernel khác nhau bắt pattern khác nhau."
)
