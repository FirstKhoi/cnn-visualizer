import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.train import MAX_TRAIN_SIZE, TrainConfig, load_digits_split, predict
from viz.matrix_view import fig_curves, fig_feature_maps
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[12]

st.set_page_config(page_title="Generalization — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="12. Generalization",
    subtitle="Train acc 100% chưa nói lên gì. Cái cần là đúng trên ảnh chưa từng thấy — và vì sao các kỹ thuật kéo đường val lên.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Mạng đủ lớn có thể <b>học thuộc</b> tập train — kể cả nhãn sai. Lúc đó train loss "
    "tiếp tục giảm nhưng val loss quay đầu tăng: mạng đang học chi tiết riêng của từng "
    "ảnh train thay vì quy luật chung. Khoảng cách train − val (gap) là thước đo overfit.",
    color=COLOR,
    label="Vì sao quan trọng",
)

OVERFIT = dict(
    train_size=300, label_noise=0.3, dropout=0.0, weight_decay=0.0, augment=False, use_bn=False, epochs=60
)
PRESETS = {
    "1. Overfit: 300 ảnh, 30% nhãn sai": OVERFIT,
    "2. + Dropout 0.5": {**OVERFIT, "dropout": 0.5},
    "3. + Weight decay 0.02": {**OVERFIT, "weight_decay": 0.02},
    "4. + Augmentation (dịch ±1px)": {**OVERFIT, "augment": True},
    "5. + BatchNorm (không phải regularizer!)": {**OVERFIT, "use_bn": True},
    "6. + Nhiều data hơn (1297 ảnh)": {**OVERFIT, "train_size": MAX_TRAIN_SIZE},
    "7. Data sạch (0% nhãn sai)": {**OVERFIT, "label_noise": 0.0},
}


def apply_preset() -> None:
    for key, value in PRESETS[st.session_state["g_preset"]].items():
        st.session_state[f"g_{key}"] = value


def run_label(cfg: TrainConfig) -> str:
    parts = [f"{cfg.train_size} ảnh", f"noise {cfg.label_noise:.0%}"]
    if cfg.dropout:
        parts.append(f"drop {cfg.dropout}")
    if cfg.weight_decay:
        parts.append(f"wd {cfg.weight_decay}")
    if cfg.augment:
        parts.append("aug")
    if cfg.use_bn:
        parts.append("BN")
    return " · ".join(parts)


if "g_train_size" not in st.session_state:
    st.session_state["g_preset"] = next(iter(PRESETS))
    apply_preset()
if "g_runs" not in st.session_state:
    st.session_state["g_runs"] = [TrainConfig(**OVERFIT)]

with st.container(border=True):
    st.selectbox("Preset (chọn lần lượt 1 → 7 rồi bấm Chạy)", list(PRESETS), key="g_preset", on_change=apply_preset)
    c1, c2, c3 = st.columns(3)
    c1.select_slider("Số ảnh train", options=[50, 100, 200, 300, 500, 800, MAX_TRAIN_SIZE], key="g_train_size")
    c1.slider("Tỉ lệ nhãn sai", 0.0, 0.5, key="g_label_noise", step=0.1)
    c2.slider("Dropout", 0.0, 0.7, key="g_dropout", step=0.1)
    c2.select_slider("Weight decay", options=[0.0, 0.001, 0.005, 0.02, 0.05], key="g_weight_decay")
    c3.toggle("Augmentation (dịch ±1px)", key="g_augment")
    c3.toggle("BatchNorm", key="g_use_bn")
    c3.slider("Số epoch", 10, 100, key="g_epochs", step=10)

    cfg = TrainConfig(**{k: st.session_state[f"g_{k}"] for k in OVERFIT})
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Chạy & thêm vào so sánh", type="primary", use_container_width=True):
        if cfg not in st.session_state["g_runs"]:
            st.session_state["g_runs"].append(cfg)
    if b2.button("Xoá hết", use_container_width=True):
        st.session_state["g_runs"] = []

runs = st.session_state["g_runs"]
if not runs:
    st.info("Chưa có lần chạy nào — chọn preset rồi bấm **Chạy**.")
    st.stop()

results = [(run_label(c), *cached_train(c)) for c in runs]

st.markdown("#### Đường học — mọi lần chạy chồng lên nhau")
c1, c2 = st.columns(2)
c1.pyplot(fig_curves([(label, h) for label, _, h in results], "acc"))
c2.pyplot(fig_curves([(label, h) for label, _, h in results], "loss"))

rows = []
for label, _, h in results:
    best = int(np.argmax(h["val_acc"]))
    rows.append(
        {
            "Lần chạy": label,
            "Train acc": h["train_acc"][-1],
            "Val acc cuối": h["val_acc"][-1],
            "Val tốt nhất": h["val_acc"][best],
            "@ epoch": best + 1,
            "Gap (train − val)": h["train_acc"][-1] - h["val_acc"][-1],
        }
    )
st.dataframe(
    pd.DataFrame(rows).style.format(
        {c: "{:.3f}" for c in ["Train acc", "Val acc cuối", "Val tốt nhất", "Gap (train − val)"]}
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Val tốt nhất ≠ val cuối nghĩa là mạng đã qua đỉnh rồi bắt đầu overfit — dừng ở epoch "
    "đó (early stopping) là kỹ thuật regularize rẻ nhất. Train acc đo trên nhãn train (có "
    "nhiễu), val đo trên nhãn sạch — nên gap ÂM là dấu hiệu tốt: mạng từ chối học thuộc nhãn sai."
)

st.markdown("#### Vì sao từng kỹ thuật kéo đường val lên")
c1, c2 = st.columns(2)
with c1:
    callout(
        "Mỗi bước train tắt ngẫu nhiên p% neuron → mạng không thể dựa vào 1 neuron cụ "
        "thể để 'nhớ' 1 ảnh, buộc thông tin phải phân tán. Giống train nhiều mạng con "
        "rồi lấy trung bình (ensemble). Train acc thấp hơn, val cao hơn.",
        color=COLOR,
        label="Dropout",
    )
    callout(
        "Cộng λ·W vào gradient → kéo mọi trọng số về 0. Học thuộc nhiễu cần trọng số lớn "
        "để uốn hàm quanh từng điểm lẻ; phạt trọng số lớn → hàm mượt hơn, bỏ qua điểm lẻ.",
        color=COLOR,
        label="Weight decay (L2)",
    )
    callout(
        "Dịch ảnh ±1px mỗi lần → mạng không bao giờ thấy lại đúng 1 ảnh, nên không thể "
        "nhớ theo pixel. Là cách 'thêm data' miễn phí, mã hoá luôn hiểu biết của ta rằng "
        "chữ số dịch 1px vẫn là chữ số đó.",
        color=COLOR,
        label="Augmentation",
    )
with c2:
    callout(
        "BN giúp tối ưu nhanh hơn (trang 9) — nhưng tối ưu nhanh hơn cũng là học thuộc "
        "nhiễu nhanh hơn. Nhiễu từ batch stats chỉ regularize rất nhẹ. Đừng kỳ vọng BN "
        "chữa overfit.",
        color=COLOR,
        label="BatchNorm",
    )
    callout(
        "Nhiều ảnh thật hơn nâng TRẦN cao nhất: quy luật chung lặp lại qua nhiều ảnh, còn "
        "nhiễu thì không (xem cột 'Val tốt nhất'). Nhưng nhìn 'Val acc cuối': train đủ lâu, "
        "mạng vẫn học thuộc được cả nhãn sai. Thêm data nâng trần; regularization và early "
        "stopping giữ mạng ở gần trần đó.",
        color=COLOR,
        label="Thêm data",
    )
    callout(
        "Nhãn sai là thứ duy nhất mạng CHỈ có thể học bằng cách học thuộc. Train acc vẫn "
        "leo lên gần 100% nghĩa là mạng đã nhớ cả nhãn sai — dấu hiệu overfit rõ nhất.",
        color=COLOR,
        label="Nhãn sai",
    )

st.markdown("#### Ảnh val bị đoán sai — lần chạy mới nhất")
label, model, _ = results[-1]
_, _, x_val, y_val = load_digits_split()
pred = predict(model, x_val).argmax(axis=1)
wrong = np.flatnonzero(pred != y_val)[:16]
st.caption(f"{label}: sai {int(np.sum(pred != y_val))}/{len(y_val)} ảnh val. 16 ảnh sai đầu tiên:")
if len(wrong):
    st.pyplot(
        fig_feature_maps(
            x_val[wrong, :, :, 0].transpose(1, 2, 0),
            titles=[f"đoán {pred[i]} · đúng {y_val[i]}" for i in wrong],
            max_cols=8,
            cmap="gray_r",
        )
    )
