import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.block import ConvBlock, make_random_kernels
from viz.matrix_view import fig_activation_hist, fig_feature_maps
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import run_or_hint

COLOR = PAGE_COLORS[7]

st.set_page_config(page_title="Conv Block — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="7. Conv Block",
    subtitle="Conv → ReLU → Pool, lặp lại — đây là 'viên gạch' xây nên mọi CNN cổ điển.",
    badge="Phase 5",
    color=COLOR,
)
callout(
    "Không train ở đây (không có backprop) — kernel sinh ngẫu nhiên chỉ để "
    "quan sát forward pass đi qua nhiều kênh trông ra sao, giống nhìn model "
    "lúc mới khởi tạo, trước khi train.",
    color=COLOR,
    label="Vì sao quan trọng",
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

with st.container(border=True):
    col_a, col_b, col_c, col_img = st.columns([1, 1, 1, 1.2])
    num_kernels = col_a.slider("Số kernel", min_value=2, max_value=8, value=4)
    kernel_size = col_b.select_slider("Kernel size", options=[3, 5], value=3)
    seed = col_c.number_input("Seed", min_value=0, value=0)
    with col_img:
        st.caption("Ảnh input (tổng hợp, 3 kênh)")
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
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
steps = run_or_hint(
    block.forward_steps,
    image,
    todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5) — cần conv2d_multichannel, relu, max_pool2d đã xong trước đó.",
)
output = steps[-1][1]

with st.container(border=True):
    st.markdown(f"#### Feature maps sau Conv → ReLU → Pool: `{output.shape}`")
    shared = st.toggle(
        "Chung thang màu cho mọi kernel",
        value=True,
        help="Tắt: mỗi map tự co giãn màu — dễ nhìn hình dạng nhưng che mất độ lớn thật.",
    )
    st.pyplot(fig_feature_maps(output, titles=[f"kernel {i}" for i in range(num_kernels)], shared_scale=shared))
    st.caption(
        f"Input {image.shape} -> {num_kernels} kernel {kernel_size}x{kernel_size} -> "
        f"output {output.shape}. Mỗi ô là 1 feature map — kernel khác nhau bắt pattern khác nhau."
    )

with st.container(border=True):
    st.markdown("#### Phân phối giá trị qua từng bước")
    st.pyplot(fig_activation_hist([("input", image)] + steps, color=COLOR))
    st.caption(
        "Sau Conv: số âm lẫn dương quanh 0. Sau ReLU: toàn bộ phần âm dồn về đúng 0 "
        "(cột cao ở 0, xem % zero). Sau Max Pool: chỉ giữ số lớn nhất mỗi vùng nên "
        "phân phối dịch sang phải, ít số 0 hơn."
    )
