import numpy as np
import streamlit as st

from cnn_core.train import TrainConfig, build_model, forward_trace, load_digits_split, param_layer_names
from viz.matrix_view import DIV, fig_curves, fig_feature_maps, fig_lines
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[11]

st.set_page_config(page_title="Backprop & Training — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="11. Backprop & Training",
    subtitle="Kernel không còn random: lan truyền ngược gradient của loss về từng lớp và sửa trọng số từng chút một.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Trang 1–8 dùng kernel random. Train = lặp lại: forward → loss → backward (chain "
    "rule: mỗi lớp nhận ∂L/∂output, trả về ∂L/∂input cho lớp trước và ∂L/∂W cho chính "
    "nó) → trừ gradient × lr vào trọng số. Lặp vài trăm bước như vậy, mạng từ đoán "
    "bừa thành đoán đúng ~99% — trang này đo xem trọng số THẬT SỰ đã đổi bao nhiêu.",
    color=COLOR,
    label="Vì sao quan trọng",
)

with st.expander("Toàn bộ vòng lặp train trong `cnn_core/train.py` (rút gọn)", expanded=False):
    st.code(
        """for xb, yb in batches:
    probs = softmax(forward(model, xb, train=True))       # forward
    dout = softmax_cross_entropy_backward(probs, yb)       # ∂L/∂logits = (p − y)/N
    for layer in reversed(model):                          # backward: chain rule
        dout = layer.backward(dout)                        #   lớp tự lưu grads["W"]
    for layer in model:                                    # SGD + momentum
        for k, param in layer.params.items():
            vel[k] = 0.9 * vel[k] - lr * layer.grads[k]
            param += vel[k]""",
        language="python",
    )

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    epochs = c1.slider("Số epoch", 1, 30, 15)
    lr = c2.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.01)
    use_bn = c3.toggle("BatchNorm", value=True)

cfg = TrainConfig(use_bn=use_bn, epochs=epochs, lr=lr)
model, history = cached_train(cfg)

final_loss = history["train_loss"][-1]
# ln(10) ≈ 2.30 = loss của việc đoán đều 10 lớp — kẹt quanh đó nghĩa là mạng không học được gì
stuck = not np.isfinite(final_loss) or final_loss > 0.9 * np.log(10)
if stuck and lr < 0.1:
    st.warning(
        f"Chưa học được gì đáng kể sau {epochs} epoch (train loss {final_loss:.2f}, chưa xuống "
        "dưới mức đoán bừa ln 10 = 2.30) — "
        f"lr = {lr} nhỏ nên mỗi bước đi rất ngắn. Tăng số epoch hoặc lr."
    )
elif stuck:
    st.error(
        f"Mạng không học được (train loss cuối {final_loss:.2f}, không tốt hơn mức đoán bừa "
        "ln 10 = 2.30) — "
        f"lr = {lr} quá lớn: mỗi bước nhảy vượt quá đáy, trọng số văng ra xa, nhiều ReLU 'chết' "
        "(luôn ra 0) và không bao giờ hồi lại. Giảm lr xuống."
    )

st.markdown("#### Đường học")
c1, c2 = st.columns(2)
c1.pyplot(fig_curves([(f"lr={lr}", history)], "loss"))
c2.pyplot(fig_curves([(f"lr={lr}", history)], "acc"))
st.caption(
    f"Val acc cuối: {history['val_acc'][-1]:.3f}. Thử lr 0.001 (học quá chậm) và 0.3 "
    "(mạng 'chết', kẹt ở đoán bừa) để thấy vì sao learning rate là tham số quan trọng nhất."
)

st.markdown("#### Gradient chảy về từng lớp")
names = param_layer_names(model)
st.pyplot(
    fig_lines(
        {name: [g[i] for g in history["grad_norms"]] for i, name in enumerate(names)},
        xlabel="epoch",
        ylabel="‖∂L/∂W‖ trung bình (log)",
        logy=True,
        figsize=(9, 3.2),
    )
)
st.caption(
    "Gradient lớn lúc đầu (mạng sai nhiều) rồi nhỏ dần khi loss giảm. Tắt BatchNorm "
    "để so độ lớn gradient giữa các lớp — lớp nào nhận gradient quá nhỏ thì học rất chậm."
)

st.markdown("#### Trước vs sau khi train")
init_model = build_model(cfg)  # cùng seed → đúng trọng số lúc khởi tạo của lần train này
with st.container(border=True):
    st.markdown("**16 kernel 3×3 của lớp Conv đầu tiên**")
    c1, c2 = st.columns(2)
    c1.caption("Lúc khởi tạo (random)")
    c1.pyplot(fig_feature_maps(init_model[0].params["W"][:, :, 0, :], max_cols=8, cmap=DIV, shared_scale=True))
    c2.caption("Sau khi train")
    c2.pyplot(fig_feature_maps(model[0].params["W"][:, :, 0, :], max_cols=8, cmap=DIV, shared_scale=True))

    w0, w1 = init_model[0].params["W"], model[0].params["W"]
    delta = w1 - w0
    steps = epochs * int(np.ceil(cfg.train_size / cfg.batch_size))
    st.markdown(f"**ΔW = W sau − W đầu** (thang màu riêng) — sau {steps} bước")
    st.pyplot(fig_feature_maps(delta[:, :, 0, :], max_cols=8, cmap=DIV, shared_scale=True))
    rel = [np.linalg.norm(layer.params["W"] - w_init.params["W"]) / np.linalg.norm(w_init.params["W"])
           for layer, w_init in zip(model, init_model) if "W" in layer.params]
    small_change = rel[0] < 0.3
    per_layer = ", ".join(f"{n} {r:.0%}" for n, r in zip(param_layer_names(model), rel))
    if small_change:
        summary = (
            f"Conv 1 chỉ đổi {rel[0]:.0%} (‖ΔW‖/‖W đầu‖) — hai lưới trên gần như giống hệt. Kernel "
            "3×3 random vốn đã là những bộ dò cạnh/độ sáng thô; phần lớn 'việc học' nằm ở các lớp "
            "sau, nơi các đặc trưng thô này được tổ hợp lại."
        )
    else:
        summary = (
            f"Conv 1 đổi {rel[0]:.0%} (‖ΔW‖/‖W đầu‖) — lr lớn / không có BN đẩy trọng số đi xa hơn "
            "nhiều. Ở cấu hình mặc định (lr 0.01, có BN) Conv 1 chỉ đổi khoảng 17%."
        )
    st.caption(f"{summary} Mức đổi từng lớp: {per_layer} — lớp càng sâu đổi càng nhiều.")

with st.container(border=True):
    _, _, x_val, y_val = load_digits_split()
    idx = st.slider("Ảnh val", 0, len(x_val) - 1, 3)
    image = x_val[idx : idx + 1]
    order = "Conv 1 → BN → ReLU" if use_bn else "Conv 1 → ReLU"
    st.markdown(f"**Output riêng của Conv 1 cho ảnh chữ số {y_val[idx]}** (trong mạng: {order})")

    def first_conv(m):
        # chỉ lấy output Conv 1 (trước BN/ReLU) để so đúng phần do KERNEL thay đổi
        return next(out for name, out in forward_trace(m, image) if name == "Conv2D")[0]

    c1, c2 = st.columns(2)
    c1.caption("Lúc khởi tạo")
    c1.pyplot(fig_feature_maps(first_conv(init_model), max_cols=8, cmap=DIV, shared_scale=True))
    c2.caption("Sau khi train")
    c2.pyplot(fig_feature_maps(first_conv(model), max_cols=8, cmap=DIV, shared_scale=True))
    if small_change:
        st.caption(
            "Kernel đổi ít nên feature map của Conv 1 trước và sau train cũng na ná nhau: mỗi map "
            "đã sáng lên ở những nét ngang / dọc / chéo nhất định ngay từ lúc random. Mạng không "
            "cần 'phát minh' bộ dò cạnh — nó học cách DÙNG chúng ở các lớp sau."
        )
    else:
        st.caption(
            "Kernel đổi nhiều nên feature map khác hẳn lúc khởi tạo. So với cấu hình mặc định "
            "(Conv 1 đổi ~17%, val ~99%): mạng không cần đổi nhiều kernel Conv 1 mới đoán tốt."
        )
