import matplotlib

matplotlib.use("Agg")
import numpy as np

from viz.matrix_view import DIV, fig_matrix
from viz.theme import TEXT


def test_fig_matrix_text_readable_on_light_cells():
    """Ô màu nhạt (giữa colormap phân kỳ) phải có chữ tối, ô đậm ở 2 đầu có chữ trắng."""
    fig = fig_matrix(np.array([[-1.0, 0.05, 1.0]]), cmap=DIV)
    colors = [t.get_color() for t in fig.axes[0].texts]
    assert colors == ["white", TEXT, "white"]


def test_fig_curves_plots_every_run_even_past_palette_size():
    from viz.matrix_view import RUN_COLORS, fig_curves

    hist = {"train_acc": [0.5, 0.9], "val_acc": [0.4, 0.8]}
    runs = [(f"run {i}", hist) for i in range(len(RUN_COLORS) + 1)]
    labels = [t.get_text() for t in fig_curves(runs).axes[0].get_legend().get_texts()]
    assert labels == [label for label, _ in runs]


def test_shared_scale_is_symmetric_around_zero_when_negative():
    from viz.matrix_view import fig_feature_maps

    fig = fig_feature_maps(np.array([[[-3.0, 1.0]]]), shared_scale=True)
    assert fig.axes[0].images[0].get_clim() == (-3.0, 3.0)
