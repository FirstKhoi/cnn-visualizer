"""Helper vẽ ma trận/feature map bằng matplotlib, dùng chung cho các trang
Streamlit. Đã viết sẵn hoàn chỉnh — không phải phần bạn cần implement.
"""

from __future__ import annotations

from itertools import cycle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from viz.theme import ACCENT, BG, BORDER, MUTED, PRIMARY, TEXT

# ---------------------------------------------------------------------------
# Colormap theo palette "Vibrant Learning" thay cho viridis/gray/RdBu mặc định
# ---------------------------------------------------------------------------

SEQ = LinearSegmentedColormap.from_list("learn_seq", ["#F1EEFD", PRIMARY])
DIV = LinearSegmentedColormap.from_list("learn_div", ["#FF7675", "#FFFFFF", "#00B894"])
WARM = LinearSegmentedColormap.from_list("learn_warm", ["#2D2A55", ACCENT, "#E17055"])

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "font.family": "sans-serif",
        "axes.edgecolor": BORDER,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
    }
)


def fig_matrix(
    matrix: np.ndarray,
    title: str | None = None,
    cmap=SEQ,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> plt.Figure:
    """Vẽ 1 ma trận 2D dạng heatmap, có số trong từng ô nếu ma trận không quá to.

    Args:
        mask: mảng bool cùng shape matrix, True = ô đã "lộ ra", False = ô
            chưa tính tới (hiện mờ + dấu '?'). Dùng cho hiển thị output đang
            được lấp dần trong các bước step-by-step. None = hiện toàn bộ.
    """
    matrix = np.asarray(matrix, dtype=float)
    h, w = matrix.shape
    if figsize is None:
        figsize = (min(1.0 + w * 0.6, 8), min(1.0 + h * 0.6, 8))
    fig, ax = plt.subplots(figsize=figsize)

    if mask is not None:
        display_matrix = np.ma.masked_where(~mask, matrix)
        cmap_masked = cmap.copy() if hasattr(cmap, "copy") else cmap
        cmap_masked.set_bad(color=BORDER)
        im = ax.imshow(display_matrix, cmap=cmap_masked)
    else:
        im = ax.imshow(matrix, cmap=cmap)

    if annotate and h * w <= 400:
        for i in range(h):
            for j in range(w):
                if mask is not None and not mask[i, j]:
                    ax.text(j, i, "?", ha="center", va="center", color=MUTED, fontsize=11, fontweight="bold")
                    continue
                value = matrix[i, j]
                # chọn màu chữ theo độ sáng thật của ô (đúng cả với colormap phân kỳ)
                r, g, b, _ = im.cmap(im.norm(value))
                color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.65 else TEXT
                ax.text(j, i, f"{value:.2g}", ha="center", va="center", color=color, fontsize=9, fontweight="medium")
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def fig_matrix_with_highlight(
    matrix: np.ndarray,
    highlight_top_left: tuple[int, int],
    highlight_size: int,
    title: str | None = None,
    cmap=SEQ,
    highlight_color: str = ACCENT,
) -> plt.Figure:
    """Giống fig_matrix nhưng vẽ thêm khung màu đánh dấu vị trí kernel/window
    hiện tại — dùng để mô phỏng kernel "trượt" qua input theo từng bước.

    Args:
        highlight_top_left: (row, col) góc trên-trái của cửa sổ đang highlight.
        highlight_size: kích thước cửa sổ (vuông).
    """
    fig = fig_matrix(matrix, title=title, cmap=cmap)
    ax = fig.axes[0]
    row, col = highlight_top_left
    # viền glow mờ phía ngoài + viền sắc bên trong, cho cảm giác "đang sáng"
    for lw, alpha in [(9, 0.25), (3, 1.0)]:
        rect = patches.Rectangle(
            (col - 0.5, row - 0.5),
            highlight_size,
            highlight_size,
            linewidth=lw,
            edgecolor=highlight_color,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)
    return fig


def fig_side_by_side(
    matrices: list[np.ndarray],
    titles: list[str] | None = None,
    cmap=SEQ,
) -> plt.Figure:
    """Vẽ nhiều ma trận cạnh nhau để so sánh (vd: input gốc vs input đã pad,
    hoặc max-pool vs avg-pool trên cùng input)."""
    n = len(matrices)
    titles = titles or [None] * n
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, matrix, title in zip(axes, matrices, titles):
        matrix = np.asarray(matrix)
        ax.imshow(matrix, cmap=cmap)
        if title:
            ax.set_title(title, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    return fig


def fig_feature_maps(
    feature_maps: np.ndarray,
    titles: list[str] | None = None,
    max_cols: int = 4,
    cmap=WARM,
    shared_scale: bool = False,
) -> plt.Figure:
    """Vẽ lưới các feature map từ 1 tensor shape (H, W, C) — mỗi kênh 1 ô.

    Dùng ở trang Conv Block / Full Pipeline để xem tất cả feature map ra từ
    1 lớp conv cùng lúc.

    shared_scale=False: mỗi map tự co giãn màu riêng -> dễ nhìn HÌNH DẠNG
        nhưng che mất ĐỘ LỚN (map toàn số ~0.001 trông sáng y như map ~10).
    shared_scale=True: mọi map chung 1 thang màu + 1 colorbar -> so được độ
        lớn giữa các kênh (thấy kênh nào "chết", kênh nào trội). Nếu có số
        âm, thang đối xứng quanh 0.
    """
    feature_maps = np.asarray(feature_maps)
    channels = feature_maps.shape[-1]
    cols = min(max_cols, channels)
    rows = int(np.ceil(channels / cols))
    vmin = vmax = None
    if shared_scale:
        vmin, vmax = float(feature_maps.min()), float(feature_maps.max())
        if vmin < 0:  # có số âm -> thang đối xứng quanh 0 để 0 luôn ở giữa colormap
            vmax = max(-vmin, vmax)
            vmin = -vmax
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(channels):
        ax = axes[c]
        im = ax.imshow(feature_maps[:, :, c], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titles[c] if titles else f"kernel {c}", fontsize=10)
        ax.axis("off")
    for c in range(channels, len(axes)):
        axes[c].axis("off")
    fig.tight_layout()
    if shared_scale:
        fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    return fig


def fig_activation_hist(
    named_arrays: list[tuple[str, np.ndarray]],
    color: str = PRIMARY,
    max_cols: int = 4,
) -> plt.Figure:
    """Histogram giá trị activation cho từng lớp, CHUNG trục x để thấy phân
    phối co lại / phình ra qua độ sâu. Tiêu đề mỗi ô ghi std và % số 0."""
    n = len(named_arrays)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.3 * rows), sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, (name, arr) in zip(axes, named_arrays):
        values = np.asarray(arr, dtype=float).ravel()
        ax.hist(values, bins=40, color=color, alpha=0.85)
        ax.set_title(f"{name}\nstd {values.std():.3g} · zero {100 * np.mean(values == 0):.0f}%", fontsize=9)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


RUN_COLORS = ["#6C5CE7", "#E17055", "#00B894", "#4D96FF", "#FD79A8", "#FDCB6E", "#636E72", "#00CEC9"]


def fig_lines(
    series: dict[str, list[float]],
    xlabels: list[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    logy: bool = False,
    figsize: tuple[float, float] = (6, 3.4),
) -> plt.Figure:
    """Nhiều đường trên 1 trục, mỗi series 1 màu. xlabels (tuỳ chọn) đặt tên
    cho từng điểm trên trục x (vd tên lớp) thay vì số 0, 1, 2..."""
    fig, ax = plt.subplots(figsize=figsize)
    for color, (name, values) in zip(cycle(RUN_COLORS), series.items()):
        ax.plot(range(len(values)), values, marker="o", markersize=4, linewidth=2, color=color, label=name)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_curves(runs: list[tuple[str, dict]], metric: str = "acc", figsize: tuple[float, float] = (6, 3.6)) -> plt.Figure:
    """Đường học train (nét đứt) vs val (nét liền) cho nhiều lần chạy chồng
    lên nhau. Chấm tròn = epoch có val tốt nhất (nơi early stopping sẽ dừng).

    runs: [(nhãn, history)] với history từ cnn_core.train.train().
    metric: "acc" hoặc "loss".
    """
    fig, ax = plt.subplots(figsize=figsize)
    for color, (label, hist) in zip(cycle(RUN_COLORS), runs):
        train_vals, val_vals = hist[f"train_{metric}"], hist[f"val_{metric}"]
        epochs = np.arange(1, len(val_vals) + 1)
        ax.plot(epochs, train_vals, linestyle="--", linewidth=1.5, color=color, alpha=0.7)
        ax.plot(epochs, val_vals, linewidth=2.2, color=color, label=label)
        best = int(np.argmax(val_vals) if metric == "acc" else np.argmin(val_vals))
        ax.scatter([epochs[best]], [val_vals[best]], color=color, s=40, zorder=3, edgecolor="white")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy" if metric == "acc" else "loss (cross-entropy)")
    ax.set_title("nét đứt = train · nét liền = val · chấm = val tốt nhất", fontsize=9, color=MUTED, fontweight="normal")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_bars(
    values: np.ndarray,
    labels: list[str] | None = None,
    highlight: int | None = None,
    title: str | None = None,
    color: str = PRIMARY,
    highlight_color: str = ACCENT,
    figsize: tuple[float, float] = (4.2, 2.8),
) -> plt.Figure:
    """Bar chart 1 vector (logits, xác suất, gradient...). highlight = index
    cột tô màu khác (vd lớp đúng). Có đường 0 để thấy rõ giá trị âm."""
    values = np.asarray(values, dtype=float)
    labels = labels if labels is not None else [str(i) for i in range(len(values))]
    colors = [highlight_color if i == highlight else color for i in range(len(values))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color=BORDER, linewidth=1)
    if len(values) <= 12:
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=TEXT)
    else:
        ax.set_xticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
