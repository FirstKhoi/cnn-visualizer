import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.layers import cross_entropy, softmax, softmax_cross_entropy_backward
from cnn_core.train import TrainConfig, forward_trace, load_digits_split
from viz.matrix_view import DIV, fig_bars, fig_feature_maps
from viz.theme import ACCENT, BORDER, PAGE_COLORS, TEXT, callout, formula_box, hero, inject_base_css
from viz.widgets import cached_train, step_controls

COLOR = PAGE_COLORS[10]

st.set_page_config(page_title="Softmax + Cross-Entropy — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="10. Softmax + Cross-Entropy",
    subtitle="Biến 10 con số thô (logits) thành xác suất, rồi đo mạng 'sai' đến mức nào bằng 1 con số duy nhất.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Lớp cuối của CNN chỉ nhả ra 10 số thực bất kỳ (logits). Softmax biến chúng "
    "thành xác suất (dương, tổng = 1). Cross-entropy = −log(xác suất gán cho lớp "
    "đúng): đoán đúng chắc chắn → loss ≈ 0, đoán sai chắc chắn → loss rất lớn. "
    "Đây là con số mà toàn bộ quá trình train cố kéo xuống.",
    color=COLOR,
    label="Vì sao quan trọng",
)

# ---------------------------------------------------------------------------
# (a) Tự kéo logits
# ---------------------------------------------------------------------------
st.markdown("#### (a) Tự kéo logits — xem xác suất, loss và gradient đổi theo")
with st.container(border=True):
    n_classes = 5
    defaults = [2.0, 1.0, 0.5, -1.0, 0.0]
    cols = st.columns(n_classes + 1)
    logits = np.array([cols[i].slider(f"z{i}", -5.0, 5.0, defaults[i], 0.1) for i in range(n_classes)])
    true_class = cols[-1].radio("Lớp đúng", list(range(n_classes)), index=0)

    probs = softmax(logits)
    loss = cross_entropy(probs[None], np.array([true_class]))
    grad = softmax_cross_entropy_backward(probs[None], np.array([true_class]))[0]
    labels = [str(i) for i in range(n_classes)]

    c1, c2, c3 = st.columns(3)
    c1.pyplot(fig_bars(logits, labels, highlight=true_class, title="Logits z (thô)", color=COLOR))
    c2.pyplot(fig_bars(probs, labels, highlight=true_class, title="Softmax p (tổng = 1)", color=COLOR))
    c3.pyplot(fig_bars(grad, labels, highlight=true_class, title="Gradient ∂L/∂z = p − y", color=COLOR))

    exps = " + ".join(f"e^{z:.1f}" for z in logits)
    formula_box(
        f"p{true_class} = e^{logits[true_class]:.1f} / ({exps}) = {probs[true_class]:.3f}<br>"
        f"L = −log p{true_class} = −log({probs[true_class]:.3f}) = {loss:.3f}",
        color=COLOR,
    )

    c_curve, c_text = st.columns([1.2, 1])
    with c_curve:
        p = np.linspace(0.005, 1, 200)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(p, -np.log(p), color=COLOR, linewidth=2.5)
        ax.scatter([probs[true_class]], [loss], color=ACCENT, s=80, zorder=3, edgecolor=TEXT)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax.set_xlabel("p (xác suất gán cho lớp đúng)")
        ax.set_ylabel("loss = −log p")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
    with c_text:
        st.markdown(
            "- Đường cong **dốc đứng khi p → 0**: đoán sai mà còn tự tin bị phạt cực nặng "
            "(p = 0.01 → loss 4.6), trong khi từ 0.9 lên 0.99 chỉ bớt được 0.09.\n"
            "- Gradient **p − y** gọn đến bất ngờ: logit lớp đúng bị đẩy *lên* một lượng "
            "(1 − p), mọi logit khác bị đẩy *xuống* đúng bằng xác suất của nó.\n"
            "- Cộng cùng 1 số vào mọi logit → softmax không đổi. Chỉ **chênh lệch** giữa "
            "các logit là quan trọng."
        )

# ---------------------------------------------------------------------------
# (b) 1 ảnh thật đi hết mạng tới loss
# ---------------------------------------------------------------------------
st.markdown("#### (b) 1 ảnh chữ số đi hết mạng đã train, tới tận loss")
with st.container(border=True):
    model, _ = cached_train(TrainConfig())
    _, _, x_val, y_val = load_digits_split()
    idx = st.slider("Chọn ảnh trong tập val", 0, len(x_val) - 1, 0)
    image, label = x_val[idx : idx + 1], int(y_val[idx])

    trace = forward_trace(model, image)
    probs = softmax(trace[-1][1])
    loss = cross_entropy(probs, np.array([label]))
    stages = trace + [("Softmax", probs), ("Cross-Entropy", np.array([[loss]]))]

    step = step_controls("ce_trace", len(stages), color=COLOR, step_label="Lớp")
    name, out = stages[step]
    st.markdown(f"**{name}** — shape `{out.shape[1:]}` · nhãn đúng: **{label}**")

    digits = [str(d) for d in range(10)]
    if name == "Input":
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(out[0, :, :, 0], cmap="gray_r")
        ax.axis("off")
        st.pyplot(fig)
    elif out.ndim == 4:
        st.pyplot(fig_feature_maps(out[0], max_cols=8, shared_scale=True, cmap=DIV if name == "Conv2D" else "magma"))
    elif name == "Cross-Entropy":
        formula_box(f"L = −log p{label} = −log({probs[0, label]:.4f}) = {loss:.4f}", color=COLOR)
    elif name == "Softmax":
        st.pyplot(fig_bars(out[0], digits, highlight=label, title="xác suất", color=COLOR, figsize=(6, 2.8)))
    elif step == len(trace) - 1:
        st.pyplot(fig_bars(out[0], digits, highlight=label, title="logits", color=COLOR, figsize=(6, 2.8)))
    else:
        st.pyplot(fig_bars(out[0], title=f"vector {out.shape[1]} chiều", color=COLOR, figsize=(6, 2.4)))

    pred = int(probs.argmax())
    st.caption(
        f"Mạng đoán **{pred}** với xác suất {probs[0, pred]:.1%} — "
        + ("đúng." if pred == label else f"SAI (đúng là {label}). Thử bấm tới bước Softmax xem mạng phân vân giữa các lớp nào.")
    )
