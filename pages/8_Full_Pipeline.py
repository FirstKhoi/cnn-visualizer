import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from cnn_core.block import ConvBlock, make_random_kernels
from viz.matrix_view import fig_feature_maps
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import run_or_hint, step_controls

COLOR = PAGE_COLORS[8]

st.set_page_config(page_title="Full Pipeline — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="8. Full Pipeline",
    subtitle="Xếp nhiều Conv Block liên tiếp và xem 1 ảnh thật đi qua toàn bộ chuỗi, từng block một.",
    badge="Phase 6",
    color=COLOR,
)
callout(
    "Đúng cách LeNet/VGG hoạt động (chỉ khác: kernel ở đây random, không "
    "train). Theo dõi shape thu nhỏ dần qua từng block, khớp với công thức "
    "đã học ở trang 4 — đây là lúc mọi khái niệm trước đó ráp lại thành 1 mạng.",
    color=COLOR,
    label="Vì sao quan trọng",
)


def make_synthetic_image(size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 3))
    img[: size // 2, : size // 2, 0] = 1.0
    img[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)
    rng = np.random.default_rng(1)
    img[:, :, 2] = rng.uniform(0, 0.3, size=(size, size))
    return img


with st.container(border=True):
    col_upload, col_size = st.columns([2, 1])
    uploaded = col_upload.file_uploader(
        "Ảnh của bạn (tuỳ chọn — không upload thì dùng ảnh tổng hợp)", type=["png", "jpg", "jpeg"]
    )
    image_size = col_size.select_slider("Resize ảnh về (vuông)", options=[32, 64], value=32)

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB").resize((image_size, image_size))
        image = np.asarray(pil_image, dtype=float) / 255.0
    else:
        image = make_synthetic_image(image_size)
        st.caption("Đang dùng ảnh tổng hợp mặc định — upload ảnh riêng ở trên để thử với dữ liệu thật.")

    col_img, col_cfg1, col_cfg2 = st.columns([1, 1, 1])
    with col_img:
        st.caption("Ảnh input")
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.imshow(image)
        ax.axis("off")
        st.pyplot(fig)
    num_blocks = col_cfg1.slider("Số block", min_value=1, max_value=3, value=2)
    kernels_per_block = col_cfg2.slider("Số kernel mỗi block", min_value=2, max_value=8, value=4)

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

# Forward toàn bộ trước (để có sẵn dữ liệu), nhưng CHỈ hiển thị tới đúng
# block mà người dùng đã "đi" tới qua step_controls bên dưới.
activations = [("input", image)]
x = image
for i, block in enumerate(blocks):
    x = run_or_hint(
        block.forward,
        x,
        todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5).",
    )
    activations.append((f"block {i + 1}", x))

st.markdown("#### Forward pass — đi qua từng block một")
with st.container(border=True):
    step = step_controls("pipeline_walk", len(activations), color=COLOR, step_label="Lớp")
    name, activation = activations[step]

    if step == 0:
        st.markdown(f"**Input** — shape `{activation.shape}`")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(activation)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.markdown(f"**Block {step}** — shape `{activation.shape}` (sau Conv → ReLU → Pool)")
        st.pyplot(fig_feature_maps(activation, titles=[f"k{c}" for c in range(activation.shape[-1])], max_cols=4))

st.divider()
st.markdown("#### Tóm tắt shape qua pipeline (tính tới bước hiện tại)")
visible = activations[: step + 1]
st.table({"Lớp": [n for n, _ in visible], "Shape": [str(a.shape) for _, a in visible]})
