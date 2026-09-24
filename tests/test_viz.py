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
