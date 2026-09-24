import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.analysis import (
    center_concentration,
    confusion_matrix,
    nearest_centroid_accuracy,
    occlusion_map,
    pca_2d,
    receptive_field,
    receptive_field_map,
    saliency_map,
)
from cnn_core.layers import Conv2D, MaxPool2D, ReLU
from cnn_core.train import TrainConfig, build_model, forward_trace, load_digits_split, predict
from viz.matrix_view import fig_feature_maps, fig_lines, fig_matrix, fig_scatter_grid
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[14]

st.set_page_config(page_title="Mạng nhìn thấy gì — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="14. Mạng nhìn thấy gì?",
    subtitle="Mỗi neuron nhìn vùng nào của ảnh, cả tập dữ liệu biến đổi ra sao qua từng lớp, và pixel nào quyết định câu trả lời.",
    badge="Phần 3 · Bản chất",
    color=COLOR,
)

# ---------------------------------------------------------------------------
# (a) Receptive field
# ---------------------------------------------------------------------------
st.markdown("#### (a) Receptive field — càng sâu, mỗi neuron nhìn vùng càng rộng")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    blocks = c1.slider("Số block Conv → ReLU → [Pool]", 1, 4, 3)
    k = c2.select_slider("Kernel", options=[3, 5], value=3)
    use_pool = c3.toggle("Max pool 2×2 sau mỗi conv", value=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 96, 96, 1))  # trung bình nhiều ảnh random cho gradient mượt
    layers, specs, ch = [], [], 1
    maps, titles = [], []
    for depth in range(1, blocks + 1):
        layers += [Conv2D(ch, 4, k, padding=k // 2, rng=rng), ReLU()]
        specs.append((k, 1))
        if use_pool:
            layers.append(MaxPool2D(2, 2))
            specs.append((2, 2))
        ch = 4
        grad = receptive_field_map(layers, x)
        rows, cols = np.nonzero(grad > 0)
        measured = int(rows.max() - rows.min() + 1)
        maps.append(grad / grad.max())
        titles.append(f"{depth} block · lý thuyết {receptive_field(specs)} · đo {measured} px")

    # cắt mọi map theo vùng của lớp sâu nhất (+ 4 px lề) để thấy vùng nhìn lớn dần
    rows, cols = np.nonzero(maps[-1] > 0)
    window = (
        slice(max(rows.min() - 4, 0), rows.max() + 5),
        slice(max(cols.min() - 4, 0), cols.max() + 5),
    )
    st.pyplot(fig_feature_maps(np.stack([m[window] for m in maps], axis=-1), titles=titles, max_cols=4))

    r = receptive_field(specs)
    share, area, edge = center_concentration(maps[-1], r)
    profile = maps[-1][rows.min() + r // 2, cols.min() : cols.min() + r]
    st.pyplot(
        fig_lines(
            {"|gradient| trên hàng qua tâm": list(profile / profile.max())},
            xlabels=[str(i - r // 2) for i in range(r)] if r <= 40 else None,
            xlabel="vị trí so với tâm (px)",
            ylabel="ảnh hưởng (chuẩn hoá)",
            figsize=(9, 3.0),
        )
    )
    spread = (
        "ảnh hưởng dồn rõ về giữa"
        if share > 1.5 * area
        else "ảnh hưởng còn khá đều — thêm block để thấy nó dồn về giữa"
    )
    st.caption(
        f"Công thức: r ← r + (k − 1)·jump, jump ← jump·stride — pool nhân đôi bước nhảy nên vùng "
        f"nhìn tăng rất nhanh. Nhưng vùng {r}×{r} không đều: hình vuông giữa chiếm {area:.0%} diện "
        f"tích mà mang {share:.0%} ảnh hưởng, mép vùng chỉ còn {edge:.0%} so với tâm — {spread}. "
        "Pixel ở giữa có nhiều đường đi tới neuron hơn pixel ở mép, nên receptive field "
        "<i>hiệu dụng</i> nhỏ hơn lý thuyết và càng sâu càng dồn về giữa (Luo et al., 2016)."
    )

# ---------------------------------------------------------------------------
# (b) Biểu diễn qua từng lớp
# ---------------------------------------------------------------------------
st.markdown("#### (b) Cả tập val đi qua từng lớp — các lớp chữ số tách nhau dần")
trained, _ = cached_train(TrainConfig())
random_net = build_model(TrainConfig())  # cùng kiến trúc, trọng số lúc khởi tạo
x_tr, y_tr, x_val, y_val = load_digits_split(500)

with st.container(border=True):

    def stages(model):
        trace_tr, trace_val = forward_trace(model, x_tr), forward_trace(model, x_val)
        names = [n for n, _ in trace_val]
        picks = [0] + [i for i, n in enumerate(names) if n == "MaxPool2D"] + [len(names) - 2, len(names) - 1]
        return [(trace_tr[i][1], trace_val[i][1]) for i in picks]

    labels = ["Pixel", "Pool 1", "Pool 2", "Dense ẩn", "Logits"]
    rows_data = {"Random": stages(random_net), "Đã train": stages(trained)}
    panels, panel_titles, accs = [], [], {}
    for row_name, feats in rows_data.items():
        accs[row_name] = []
        for label, (f_tr, f_val) in zip(labels, feats):
            acc = nearest_centroid_accuracy(f_tr, y_tr, f_val, y_val)
            accs[row_name].append(acc)
            panels.append(pca_2d(f_val))
            panel_titles.append(f"{row_name} · {label} ({f_val[0].size} chiều)\ntách lớp {acc:.0%}")
    st.pyplot(fig_scatter_grid(panels, y_val, panel_titles, rows=2))
    st.pyplot(fig_lines(accs, xlabels=labels, ylabel="nearest-centroid acc", figsize=(9, 3.0)))
    st.caption(
        f"Mỗi điểm là 1 ảnh val, chiếu PCA xuống 2 chiều, màu = chữ số thật. 'Tách lớp' = gán mỗi "
        f"ảnh cho lớp có tâm gần nhất trong không gian ĐẦY ĐỦ của lớp đó. Mạng đã train: "
        f"{accs['Đã train'][0]:.0%} → {accs['Đã train'][-1]:.0%}. Mạng random: "
        f"{accs['Random'][0]:.0%} → {accs['Random'][-1]:.0%}."
    )
    callout(
        "Mỗi lớp là 1 phép biến đổi không gian. Mạng đã train uốn dữ liệu sao cho ảnh cùng "
        "chữ số co cụm lại, khác chữ số đẩy ra xa — tới logits thì chỉ cần 1 đường thẳng là "
        "tách được. Mạng random cũng biến đổi dữ liệu, nhưng không có mục tiêu nên cấu trúc "
        "ban đầu còn bị xoá dần. PCA chỉ giữ 2 trong nhiều chiều nên cụm trông chồng lên nhau "
        "hơn thực tế — tin con số 'tách lớp' hơn hình.",
        color=COLOR,
        label="Đọc hình",
    )

# ---------------------------------------------------------------------------
# (c) Pixel nào quyết định
# ---------------------------------------------------------------------------
st.markdown("#### (c) Pixel nào quyết định dự đoán")
MODELS = {
    "Mặc định (1297 ảnh sạch)": TrainConfig(),
    "Overfit (300 ảnh, 30% nhãn sai)": TrainConfig(train_size=300, label_noise=0.3, use_bn=False, epochs=60),
}
with st.container(border=True):
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Mô hình", list(MODELS))
    model, _ = cached_train(MODELS[choice])
    probs_all = predict(model, x_val)
    pred_all = probs_all.argmax(axis=1)
    wrong = np.flatnonzero(pred_all != y_val)
    idx = c2.slider("Ảnh val", 0, len(x_val) - 1, int(wrong[0]) if len(wrong) else 0)
    image, label = x_val[idx], int(y_val[idx])
    pred = int(pred_all[idx])

    occ = occlusion_map(model, image, label, patch=2)
    sal = saliency_map(model, image, label)
    p1, p2, p3 = st.columns(3)
    with p1:
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(image[:, :, 0], cmap="gray_r")
        ax.set_title(f"đúng {label} · đoán {pred} ({probs_all[idx, pred]:.0%})", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
    p2.pyplot(fig_feature_maps(occ[:, :, None], titles=[f"Occlusion: che 2×2 → P({label}) rơi"], max_cols=1, shared_scale=True))
    p3.pyplot(fig_feature_maps(sal[:, :, None], titles=[f"Saliency |∂logit {label} / ∂pixel|"], max_cols=1))
    st.caption(
        "Occlusion: vùng đậm = thiếu nó thì mạng mất tự tin vào lớp đúng (số âm = che đi lại "
        "tự tin hơn). Saliency: pixel nào chỉ cần nhích 1 chút là logit đổi nhiều. Hai cách hỏi "
        "khác nhau nên không nhất thiết trùng. Mặc định chọn sẵn 1 ảnh bị đoán sai (nếu có)."
    )

    cm = confusion_matrix(y_val, pred_all)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:3]
    c_cm, c_txt = st.columns([1.3, 1])
    c_cm.pyplot(fig_matrix(cm, title="Nhãn đúng (hàng) × dự đoán (cột)", figsize=(5.2, 5.2)))
    with c_txt:
        st.metric("Val acc", f"{np.mean(pred_all == y_val):.1%}", help=f"{len(wrong)} / {len(y_val)} ảnh sai")
        if top:
            st.markdown("**Nhầm nhiều nhất**\n" + "\n".join(f"- {n} ảnh **{i}** bị đoán là **{j}**" for n, i, j in top))
        else:
            st.markdown("Không nhầm ảnh nào.")
        st.caption("Đổi sang mô hình overfit ở trên để so: nó nhầm nhiều hơn hẳn — và nhầm ở những cặp nào.")
