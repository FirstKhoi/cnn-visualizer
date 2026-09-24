import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.layers import BatchNorm2D, Conv2D, ReLU
from cnn_core.train import TrainConfig, load_digits_split
from viz.matrix_view import DIV, fig_activation_hist, fig_curves, fig_lines, fig_matrix
from viz.theme import PAGE_COLORS, callout, formula_box, hero, inject_base_css
from viz.widgets import cached_train, editable_matrix

COLOR = PAGE_COLORS[9]

st.set_page_config(page_title="BatchNorm — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="9. Batch Normalization",
    subtitle="Ép mỗi kênh về mean 0 / std 1 trên từng mini-batch — để tín hiệu không tắt dần hay nổ tung qua độ sâu.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Trang 8 cho thấy std activation co lại ~½ sau mỗi block. Khi train, trọng số "
    "mỗi lớp thay đổi liên tục nên phân phối input của lớp sau cũng trôi theo — lớp "
    "sau phải liên tục thích nghi lại. BatchNorm chuẩn hoá lại phân phối đó sau "
    "mỗi lớp, rồi cho mạng tự học scale (γ) và shift (β) phù hợp.",
    color=COLOR,
    label="Vì sao quan trọng",
)

x_train, _, _, _ = load_digits_split()

# ---------------------------------------------------------------------------
# (a) Từng bước tính trên 1 mini-batch nhỏ
# ---------------------------------------------------------------------------
st.markdown("#### (a) BN tính gì — trên 1 mini-batch 6 mẫu × 3 kênh")
with st.container(border=True):
    rng = np.random.default_rng(0)
    default_batch = np.round(
        np.column_stack([rng.normal(5, 2, 6), rng.normal(-3, 0.5, 6), rng.normal(0, 10, 6)]), 1
    )
    c_in, c_ctrl = st.columns([2, 1])
    with c_in:
        st.caption("Mỗi hàng = 1 mẫu, mỗi cột = 1 kênh. 3 kênh cố ý có thang đo rất khác nhau — sửa tuỳ ý.")
        batch = editable_matrix("bn_batch", default_batch)
    with c_ctrl:
        gamma = st.slider("γ (scale học được)", 0.1, 3.0, 1.0, 0.1)
        beta = st.slider("β (shift học được)", -3.0, 3.0, 0.0, 0.1)

    bn = BatchNorm2D(3)
    bn.params["gamma"][:] = gamma
    bn.params["beta"][:] = beta
    out = bn.forward(batch, train=True)
    mean, var = batch.mean(axis=0), batch.var(axis=0)
    x_hat = (batch - mean) / np.sqrt(var + bn.eps)

    st.table(
        pd.DataFrame(
            {"μ (mean kênh)": mean, "σ² (var kênh)": var, "mean sau BN": out.mean(axis=0), "std sau BN": out.std(axis=0)},
            index=[f"kênh {c}" for c in range(3)],
        ).round(3)
    )
    formula_box(
        f"x̂ = (x − μ) / √(σ² + ε) &nbsp;&nbsp;→&nbsp;&nbsp; ô [0, 0]: ({batch[0, 0]:.2f} − {mean[0]:.2f}) / "
        f"√({var[0]:.2f} + 1e-5) = {x_hat[0, 0]:.3f}<br>"
        f"y = γ·x̂ + β = {gamma:.1f}·{x_hat[0, 0]:.3f} + {beta:.1f} = {out[0, 0]:.3f}",
        color=COLOR,
    )
    c1, c2, c3 = st.columns(3)
    c1.pyplot(fig_matrix(batch, title="x (input)", cmap=DIV))
    c2.pyplot(fig_matrix(x_hat, title="x̂ (chuẩn hoá)", cmap=DIV))
    c3.pyplot(fig_matrix(out, title="y = γx̂ + β", cmap=DIV))
    st.caption(
        "Sau BN mỗi cột có mean = β và std = γ, bất kể thang đo ban đầu. Nếu γ = σ và β = μ "
        "thì BN trả lại đúng input — nghĩa là BN không làm mất khả năng biểu diễn, chỉ cho "
        "mạng một cách 'dễ điều khiển' hơn để chọn phân phối."
    )

# ---------------------------------------------------------------------------
# (b) Stack nhiều conv random: có vs không BN
# ---------------------------------------------------------------------------
st.markdown("#### (b) Vì sao cần — stack nhiều lớp conv random, có vs không BN")
with st.container(border=True):
    c1, c2 = st.columns(2)
    depth = c1.slider("Số lớp conv", 2, 10, 6)
    init_name = c2.selectbox(
        "Khởi tạo trọng số",
        ["Nhỏ (He × 0.5)", "He (chuẩn)", "Lớn (He × 2)"],
        help="He init = std √(2/fan_in), thiết kế để giữ std ổn định qua ReLU.",
    )
    scale = {"Nhỏ (He × 0.5)": 0.5, "He (chuẩn)": 1.0, "Lớn (He × 2)": 2.0}[init_name]

    rng = np.random.default_rng(1)
    convs = []
    in_ch = 1
    for _ in range(depth):
        conv = Conv2D(in_ch, 8, 3, padding=1, rng=rng)
        conv.params["W"] *= scale
        convs.append(conv)
        in_ch = 8

    batch_imgs = x_train[:64]
    results = {}
    for use_bn in (False, True):
        x, per_layer = batch_imgs, []
        for i, conv in enumerate(convs):
            x = conv.forward(x)
            if use_bn:
                x = BatchNorm2D(8).forward(x, train=True)
            x = ReLU().forward(x)
            per_layer.append((f"lớp {i + 1}", x))
        results["Có BN" if use_bn else "Không BN"] = per_layer

    names = [n for n, _ in results["Không BN"]]
    st.pyplot(
        fig_lines(
            {label: [float(a.std()) for _, a in layers] for label, layers in results.items()},
            xlabels=names,
            ylabel="std activation (log)",
            logy=True,
            figsize=(9, 3.2),
        )
    )
    tab_off, tab_on = st.tabs(["Histogram — không BN", "Histogram — có BN"])
    with tab_off:
        st.pyplot(fig_activation_hist(results["Không BN"], color="#E17055", max_cols=5))
    with tab_on:
        st.pyplot(fig_activation_hist(results["Có BN"], color=COLOR, max_cols=5))
    st.caption(
        "Không BN: std nhân lên đều đặn qua mỗi lớp (×0.5 → co về ~0, ×2 → nổ), nên sau "
        "vài chục lớp tín hiệu và gradient đều hỏng. He init đúng giữ được std lúc khởi "
        "tạo, nhưng trong lúc train trọng số trôi đi. Có BN: std mỗi lớp luôn được kéo về "
        "cùng mức, bất kể khởi tạo."
    )

# ---------------------------------------------------------------------------
# (c) Train vs eval
# ---------------------------------------------------------------------------
st.markdown("#### (c) Train mode vs eval mode — batch stats vs running stats")
with st.container(border=True):
    batch_size = st.select_slider("Batch size", options=[2, 4, 8, 16, 32, 64, 128], value=8)
    conv = Conv2D(1, 4, 3, padding=1, rng=np.random.default_rng(2))
    feats = conv.forward(x_train)  # (N, 8, 8, 4)
    population_mean = float(feats[..., 0].mean())

    bn = BatchNorm2D(4)
    rng = np.random.default_rng(3)
    batch_means, running = [], []
    for _ in range(60):
        idx = rng.choice(len(feats), batch_size, replace=False)
        batch_means.append(float(feats[idx][..., 0].mean()))
        bn.forward(feats[idx], train=True)
        running.append(float(bn.running_mean[0]))

    st.pyplot(
        fig_lines(
            {"μ của từng batch": batch_means, "running mean": running, "mean toàn bộ data": [population_mean] * 60},
            xlabel="batch thứ",
            ylabel="mean kênh 0",
            figsize=(9, 3.2),
        )
    )
    st.caption(
        f"Batch nhỏ → μ mỗi batch dao động mạnh quanh mean thật (độ lệch ∝ 1/√batch). "
        f"Running mean = trung bình trượt (momentum {bn.momentum}) của các μ đó, hội tụ về "
        f"mean thật. Nhiễu batch này cũng là lý do BN có chút tác dụng regularize."
    )

    image = x_train[:1]
    others_a, others_b = x_train[10:17], x_train[100:107]
    bn_trained = BatchNorm2D(4)
    for i in range(0, len(feats), 64):
        bn_trained.forward(feats[i : i + 64], train=True)
    f_img, f_a, f_b = conv.forward(image), conv.forward(others_a), conv.forward(others_b)
    out_a = bn_trained.forward(np.concatenate([f_img, f_a]), train=True)[0]
    out_b = bn_trained.forward(np.concatenate([f_img, f_b]), train=True)[0]
    eval_a = bn_trained.forward(np.concatenate([f_img, f_a]), train=False)[0]
    eval_b = bn_trained.forward(np.concatenate([f_img, f_b]), train=False)[0]
    c1, c2 = st.columns(2)
    c1.metric("Train mode: cùng 1 ảnh, 2 batch khác nhau — chênh lệch output", f"{np.abs(out_a - out_b).max():.3f}")
    c2.metric("Eval mode: cùng 1 ảnh, 2 batch khác nhau — chênh lệch output", f"{np.abs(eval_a - eval_b).max():.3f}")
    callout(
        "Ở train mode, output của 1 ảnh phụ thuộc vào các ảnh <i>khác</i> cùng batch. Lúc "
        "dự đoán, 1 ảnh phải luôn cho cùng 1 kết quả (và có khi chỉ có 1 ảnh) — nên eval "
        "dùng running stats cố định đã tích luỹ lúc train. Quên chuyển sang eval mode là "
        "bug kinh điển (<code>model.eval()</code> trong PyTorch).",
        color=COLOR,
        label="Hệ quả",
    )

# ---------------------------------------------------------------------------
# (d) Hiệu quả khi train thật
# ---------------------------------------------------------------------------
st.markdown("#### (d) Khi train thật — BN giúp tối ưu nhanh hơn")
with st.container(border=True):
    epochs = st.slider("Số epoch", 3, 20, 10, key="bn_epochs")
    runs = []
    for use_bn in (False, True):
        _, history = cached_train(TrainConfig(use_bn=use_bn, epochs=epochs))
        runs.append(("Có BN" if use_bn else "Không BN", history))
    c1, c2 = st.columns(2)
    c1.pyplot(fig_curves(runs, "acc"))
    c2.pyplot(fig_curves(runs, "loss"))
    st.caption(
        f"Cùng kiến trúc, cùng khởi tạo, cùng lr. Sau epoch 1: val acc "
        f"{runs[0][1]['val_acc'][0]:.2f} (không BN) vs {runs[1][1]['val_acc'][0]:.2f} (có BN). "
        f"BN chủ yếu giúp TỐI ƯU dễ hơn — tác dụng chống overfit của nó nhỏ (xem trang 12)."
    )
