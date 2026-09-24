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


def test_feature_maps_pick_diverging_colormap_only_for_signed_data():
    from viz.matrix_view import WARM, fig_feature_maps

    signed = fig_feature_maps(np.array([[[-1.0, 2.0]]]), shared_scale=True)
    positive = fig_feature_maps(np.array([[[0.0, 2.0]]]), shared_scale=True)
    assert signed.axes[0].images[0].get_cmap().name == DIV.name
    assert positive.axes[0].images[0].get_cmap().name == WARM.name


def test_scatter_grid_draws_every_panel_with_class_colors():
    from viz.matrix_view import fig_scatter_grid

    labels = np.array([0, 1, 2, 0])
    panels = [np.random.default_rng(i).normal(size=(4, 2)) for i in range(5)]
    fig = fig_scatter_grid(panels, labels, [f"p{i}" for i in range(5)], rows=2)
    drawn = [ax for ax in fig.axes if ax.collections]
    assert len(drawn) == 5
    assert all(len(ax.collections) == 3 for ax in drawn)  # 1 nhóm điểm mỗi lớp
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["0", "1", "2"]
