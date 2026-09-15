"""Helper vẽ ma trận/feature map bằng matplotlib, dùng chung cho các trang
Streamlit. Đã viết sẵn hoàn chỉnh — không phải phần bạn cần implement.
"""

from __future__ import annotations

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
        vmin, vmax = float(np.min(matrix)), float(np.max(matrix))
        mid = (vmin + vmax) / 2 if vmax != vmin else vmin
        for i in range(h):
            for j in range(w):
                if mask is not None and not mask[i, j]:
                    ax.text(j, i, "?", ha="center", va="center", color=MUTED, fontsize=11, fontweight="bold")
                    continue
                value = matrix[i, j]
                color = "white" if value > mid else TEXT
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
) -> plt.Figure:
    """Vẽ lưới các feature map từ 1 tensor shape (H, W, C) — mỗi kênh 1 ô.

    Dùng ở trang Conv Block / Full Pipeline để xem tất cả feature map ra từ
    1 lớp conv cùng lúc.
    """
    feature_maps = np.asarray(feature_maps)
    channels = feature_maps.shape[-1]
    cols = min(max_cols, channels)
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(channels):
        ax = axes[c]
        ax.imshow(feature_maps[:, :, c], cmap=cmap)
        ax.set_title(titles[c] if titles else f"kernel {c}", fontsize=10)
        ax.axis("off")
    for c in range(channels, len(axes)):
        axes[c].axis("off")
    fig.tight_layout()
    return fig
