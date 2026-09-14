"""Helper vẽ ma trận/feature map bằng matplotlib, dùng chung cho các trang
Streamlit. Đã viết sẵn hoàn chỉnh — không phải phần bạn cần implement.
"""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def fig_matrix(
    matrix: np.ndarray,
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Vẽ 1 ma trận 2D dạng heatmap, có số trong từng ô nếu ma trận không quá to."""
    matrix = np.asarray(matrix)
    h, w = matrix.shape
    if figsize is None:
        figsize = (min(1.0 + w * 0.6, 8), min(1.0 + h * 0.6, 8))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap)
    if annotate and h * w <= 400:
        vmin, vmax = float(np.min(matrix)), float(np.max(matrix))
        mid = (vmin + vmax) / 2 if vmax != vmin else vmin
        for i in range(h):
            for j in range(w):
                value = matrix[i, j]
                color = "white" if value < mid else "black"
                ax.text(j, i, f"{value:.2g}", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_matrix_with_highlight(
    matrix: np.ndarray,
    highlight_top_left: tuple[int, int],
    highlight_size: int,
    title: str | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Giống fig_matrix nhưng vẽ thêm khung đỏ đánh dấu vị trí kernel/window
    hiện tại — dùng để mô phỏng kernel "trượt" qua input theo từng bước.

    Args:
        highlight_top_left: (row, col) góc trên-trái của cửa sổ đang highlight.
        highlight_size: kích thước cửa sổ (vuông).
    """
    fig = fig_matrix(matrix, title=title, cmap=cmap)
    ax = fig.axes[0]
    row, col = highlight_top_left
    rect = patches.Rectangle(
        (col - 0.5, row - 0.5),
        highlight_size,
        highlight_size,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    return fig


def fig_side_by_side(
    matrices: list[np.ndarray],
    titles: list[str] | None = None,
    cmap: str = "viridis",
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
        im = ax.imshow(matrix, cmap=cmap)
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_feature_maps(
    feature_maps: np.ndarray,
    titles: list[str] | None = None,
    max_cols: int = 4,
    cmap: str = "gray",
) -> plt.Figure:
    """Vẽ lưới các feature map từ 1 tensor shape (H, W, C) — mỗi kênh 1 ô.

    Dùng ở trang Conv Block / Full Pipeline để xem tất cả feature map ra từ
    1 lớp conv cùng lúc.
    """
    feature_maps = np.asarray(feature_maps)
    channels = feature_maps.shape[-1]
    cols = min(max_cols, channels)
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(channels):
        ax = axes[c]
        ax.imshow(feature_maps[:, :, c], cmap=cmap)
        ax.set_title(titles[c] if titles else f"kernel {c}")
        ax.axis("off")
    for c in range(channels, len(axes)):
        axes[c].axis("off")
    fig.tight_layout()
    return fig
