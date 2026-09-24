import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.layers import MaxPool2D
from cnn_core.shapes import conv_output_shape
from cnn_core.train import (
    MAX_TRAIN_SIZE,
    TrainConfig,
    build_model,
    count_params,
    evaluate,
    load_digits_split,
    shift_images,
)
from viz.matrix_view import fig_feature_maps, fig_lines
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[13]

st.set_page_config(page_title="Vì sao là CNN — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="13. Vì sao là CNN?",
    subtitle="Một lớp Dense cũng nhân-cộng được mọi pixel. Cái làm CNN khác là 3 giả định về ảnh — và chúng đáng giá bao nhiêu.",
    badge="Phần 3 · Bản chất",
    color=COLOR,
)
callout(
    "CNN = MLP bị <b>ràng buộc</b> theo 3 giả định về ảnh: (1) <b>locality</b> — 1 "
    "pixel chủ yếu liên quan tới hàng xóm của nó; (2) <b>weight sharing</b> — 1 "
    "pattern (cạnh, góc) có ý nghĩa như nhau ở mọi vị trí; (3) hệ quả của (2): "
    "<b>equivariance</b> — dịch ảnh thì feature map dịch theo. Ràng buộc đúng = bớt "
    "thứ phải học.",
    color=COLOR,
    label="Ý chính",
)

# ---------------------------------------------------------------------------
# (a) Đếm tham số
# ---------------------------------------------------------------------------
st.markdown("#### (a) Locality + weight sharing — đếm tham số")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    size = c1.select_slider("Ảnh vào (H = W)", options=[8, 28, 64, 224], value=28)
    c_in = c2.select_slider("Kênh vào", options=[1, 3], value=1)
    c_out = c3.select_slider("Số kernel (kênh ra)", options=[4, 8, 16, 32, 64], value=8)
    k = c4.select_slider("Kernel", options=[3, 5], value=3)

    h_out = conv_output_shape(size, k, stride=1, padding=0)
    conv_params = k * k * c_in * c_out + c_out
    n_in, n_out = size * size * c_in, h_out * h_out * c_out
    dense_params = n_in * n_out + n_out

    m1, m2, m3 = st.columns(3)
    m1.metric("Conv — tham số", f"{conv_params:,}")
    m2.metric("Dense cùng shape — tham số", f"{dense_params:,}")
    m3.metric("Dense / Conv", f"{dense_params / conv_params:,.0f}×")
    st.table(
        pd.DataFrame(
            {
                "Mỗi output nhìn bao nhiêu input": [f"{k}×{k}×{c_in} = {k * k * c_in}", f"{size}×{size}×{c_in} = {n_in:,}"],
                "Số bộ trọng số riêng": [f"{c_out} (dùng chung cho {h_out}×{h_out} vị trí)", f"{n_out:,} (mỗi output 1 bộ)"],
                "Bộ nhớ float32": [f"{conv_params * 4 / 1e3:,.1f} KB", f"{dense_params * 4 / 1e9:,.2f} GB"],
            },
            index=["Conv", "Dense"],
        )
    )
    st.caption(
        f"Input {size}×{size}×{c_in} → output {h_out}×{h_out}×{c_out} (H_out = {size} − {k} + 1 = {h_out}, "
        "công thức trang 4). Tham số của conv KHÔNG phụ thuộc kích thước ảnh; của Dense tăng theo "
        "H⁴ — thử 224×224×3."
    )

# ---------------------------------------------------------------------------
# (b) Equivariance
# ---------------------------------------------------------------------------
st.markdown("#### (b) Equivariance — dịch ảnh thì feature map dịch theo")
model, _ = cached_train(TrainConfig())
conv1 = model[0]  # Conv 1 đã train: 16 kernel 3×3, padding 1
_, _, x_val, y_val = load_digits_split()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    idx = c1.slider("Ảnh val", 0, len(x_val) - 1, 0)
    dy = c2.slider("Dịch xuống (px)", -3, 3, 2)
    dx = c3.slider("Dịch sang phải (px)", -3, 3, 1)
    channel = c4.slider("Kernel xem", 0, conv1.params["W"].shape[-1] - 1, 0)

    image = np.kron(x_val[idx, :, :, 0], np.ones((2, 2)))[None, :, :, None]  # phóng 8×8 → 16×16
    moved = shift_images(image, dy, dx)
    f, f_moved = conv1.forward(image, train=False), conv1.forward(moved, train=False)
    diff = shift_images(f, dy, dx) - f_moved
    m = max(abs(dy), abs(dx)) + 1  # bỏ dải viền: ở đó padding / phần bị cắt làm 2 bên khác nhau
    interior_err = float(np.abs(diff[:, m:-m, m:-m]).max())

    st.pyplot(
        fig_feature_maps(
            np.stack([image[0, :, :, 0], moved[0, :, :, 0]], axis=-1),
            titles=["ảnh x", f"ảnh dịch ({dy}, {dx})"],
            max_cols=2,
            cmap="gray_r",
        )
    )
    st.pyplot(
        fig_feature_maps(
            np.stack([f[0, :, :, channel], f_moved[0, :, :, channel], diff[0, :, :, channel]], axis=-1),
            titles=["conv(x)", "conv(dịch x)", "dịch(conv x) − conv(dịch x)"],
            max_cols=3,
            shared_scale=True,
        )
    )
    st.caption(
        f"Sai khác lớn nhất ở vùng trong (bỏ {m} px viền): {interior_err:.1e} — đúng bằng 0 (sai số float). "
        "Chỉ viền khác nhau, vì phần ảnh trượt ra ngoài bị cắt và padding điền 0."
    )

    pool = MaxPool2D(2, 2)
    shifts = [0, 1, 2, 3, 4]
    conv_err, pool_err = [], []
    for s in shifts:
        fs = conv1.forward(shift_images(image, 0, s), train=False)
        crop = slice(s + 1, -(s + 1))
        ref = shift_images(f, 0, s)
        conv_err.append(float(np.linalg.norm((ref - fs)[:, crop, crop]) / np.linalg.norm(f[:, crop, crop])))
        p0, ps = pool.forward(f), pool.forward(fs)
        pool_err.append(float(np.linalg.norm(ps - p0) / np.linalg.norm(p0)))
    st.pyplot(
        fig_lines(
            {
                "conv: ‖dịch(f(x)) − f(dịch x)‖ / ‖f‖ (equivariance)": conv_err,
                "sau max pool: ‖pool(f(dịch x)) − pool(f(x))‖ / ‖pool‖ (invariance)": pool_err,
            },
            xlabels=[f"dịch {s}px" for s in shifts],
            ylabel="sai khác tương đối",
            figsize=(9, 3.2),
        )
    )
    st.caption(
        f"Conv luôn equivariant (đường dưới ≈ 0). Max pool 2×2 chỉ cho bất biến MỘT PHẦN: dịch 1px "
        f"đã làm map sau pool đổi {pool_err[1]:.0%}, dịch 4px đổi {pool_err[4]:.0%}. CNN không tự "
        "động bất biến với dịch chuyển — phần (c) đo hậu quả."
    )

# ---------------------------------------------------------------------------
# (c) CNN vs MLP
# ---------------------------------------------------------------------------
st.markdown("#### (c) Train CNN vs MLP trên cùng dữ liệu")
EPOCHS = {50: 60, 100: 40, 300: 30, MAX_TRAIN_SIZE: 15}  # ít ảnh → nhiều epoch hơn cho đủ số bước
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

with st.container(border=True):
    use_bn = st.toggle("BatchNorm (cả hai mạng)", value=True)
    rows = []
    for n, epochs in EPOCHS.items():
        row = {"Số ảnh train": n}
        for arch in ("cnn", "mlp"):
            net, history = cached_train(TrainConfig(arch=arch, train_size=n, epochs=epochs, use_bn=use_bn))
            shifted = np.mean([evaluate(net, shift_images(x_val, a, b), y_val)[1] for a, b in DIRECTIONS])
            row[f"{arch.upper()} — val"] = history["val_acc"][-1]
            row[f"{arch.upper()} — val dịch 1px"] = float(shifted)
        rows.append(row)
    table = pd.DataFrame(rows)

    cnn_params = count_params(build_model(TrainConfig(use_bn=use_bn)))
    mlp_params = count_params(build_model(TrainConfig(arch="mlp", use_bn=use_bn)))
    m1, m2 = st.columns(2)
    m1.metric("CNN — tham số", f"{cnn_params:,}")
    m2.metric("MLP — tham số", f"{mlp_params:,}", help="Cố ý cho MLP nhiều tham số hơn để so sánh công bằng với nó.")

    sizes = [str(n) for n in EPOCHS]
    c1, c2 = st.columns(2)
    c1.pyplot(
        fig_lines(
            {"CNN": list(table["CNN — val"]), "MLP": list(table["MLP — val"])},
            xlabels=sizes,
            xlabel="số ảnh train",
            ylabel="val acc — ảnh gốc",
        )
    )
    c2.pyplot(
        fig_lines(
            {"CNN": list(table["CNN — val dịch 1px"]), "MLP": list(table["MLP — val dịch 1px"])},
            xlabels=sizes,
            xlabel="số ảnh train",
            ylabel="val acc — ảnh dịch 1px",
        )
    )
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c != "Số ảnh train"}), hide_index=True)

    first, last = table.iloc[0], table.iloc[-1]
    gap_small = first["CNN — val"] - first["MLP — val"]
    gap_full = last["CNN — val"] - last["MLP — val"]
    gap_shift = last["CNN — val dịch 1px"] - last["MLP — val dịch 1px"]
    st.caption(
        f"Ảnh gốc: CNN − MLP = {gap_small:+.1%} với {int(first['Số ảnh train'])} ảnh, {gap_full:+.1%} với "
        f"{int(last['Số ảnh train'])} ảnh. Ảnh dịch 1px: {gap_shift:+.1%} "
        f"(CNN {last['CNN — val dịch 1px']:.0%} vs MLP {last['MLP — val dịch 1px']:.0%}). "
        f"CNN dùng ít hơn {mlp_params - cnn_params:,} tham số."
    )
    callout(
        "Digits của sklearn đã được <b>căn giữa và chuẩn hoá</b> sẵn, nên trên ảnh gốc MLP "
        "gần theo kịp — ở đây locality không cứu được nhiều vì ảnh chỉ có 64 pixel. Lợi thế "
        "thật của CNN hiện ở 2 chỗ: <b>ít tham số hơn</b> cho cùng độ chính xác, và <b>chịu dịch "
        "chuyển tốt hơn hẳn</b> vì mỗi kernel đã thấy pattern ở mọi vị trí. Nhưng CNN cũng rơi "
        "mạnh khi dịch: 1px là 12% bề rộng ảnh 8×8, max pool chỉ bất biến cục bộ, còn lớp Dense "
        "cuối vẫn nhìn vị trí tuyệt đối. Vì vậy vẫn cần augmentation (trang 12); CNN hiện đại "
        "thay Dense cuối bằng global average pooling.",
        color=COLOR,
        label="Đọc kết quả cho đúng",
    )
