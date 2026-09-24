import numpy as np
import pytest

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
from cnn_core.train import TrainConfig, build_model, load_digits_split, train

BLOCK = [(3, 1), (2, 2)]  # Conv3×3 stride 1, rồi MaxPool 2×2 stride 2


@pytest.mark.parametrize(
    "specs,expected",
    [
        ([(3, 1)], 3),
        (BLOCK, 4),
        (BLOCK * 2, 10),
        (BLOCK * 3, 22),
        ([(3, 1)] * 3, 7),  # 3 conv 3×3 liên tiếp = 1 conv 7×7 về vùng nhìn
        ([(5, 2), (3, 1)], 9),  # stride 2 làm bước của lớp sau dài gấp đôi
    ],
)
def test_receptive_field_formula(specs, expected):
    assert receptive_field(specs) == expected


@pytest.mark.parametrize("blocks,expected", [(1, 4), (2, 10), (3, 22)])
def test_receptive_field_map_matches_formula(blocks, expected):
    rng = np.random.default_rng(0)
    layers, ch = [], 1
    for _ in range(blocks):
        layers += [Conv2D(ch, 4, 3, padding=1, rng=rng), ReLU(), MaxPool2D(2, 2)]
        ch = 4
    grad = receptive_field_map(layers, rng.normal(size=(16, 48, 48, 1)))
    rows, cols = np.nonzero(grad > 0)
    assert rows.max() - rows.min() + 1 == expected
    assert cols.max() - cols.min() + 1 == expected


def test_pca_2d_first_axis_follows_largest_variance():
    rng = np.random.default_rng(1)
    f = np.column_stack([rng.normal(0, 10, 200), rng.normal(0, 1, 200), rng.normal(0, 0.1, 200)])
    z = pca_2d(f)
    assert z.shape == (200, 2)
    assert abs(np.corrcoef(z[:, 0], f[:, 0])[0, 1]) > 0.99
    assert z[:, 0].var() > z[:, 1].var()


def test_nearest_centroid_accuracy_on_separated_and_mixed_clusters():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 30)
    separated = np.eye(3)[y] * 10 + rng.normal(0, 0.1, (90, 3))
    assert nearest_centroid_accuracy(separated, y, separated, y) == 1.0
    noise = rng.normal(size=(90, 3))
    assert nearest_centroid_accuracy(noise, y, rng.normal(size=(90, 3)), y) < 0.6


@pytest.fixture(scope="module")
def trained():
    model, _ = train(TrainConfig(epochs=5))
    _, _, x_val, y_val = load_digits_split()
    return model, x_val, y_val


def test_occlusion_map_blank_corner_does_not_matter(trained):
    model, x_val, y_val = trained
    image = x_val[0].copy()
    image[:2, :2] = 0.0  # góc trên-trái vốn trống -> che nó không đổi gì
    heat = occlusion_map(model, image, int(y_val[0]), patch=2)
    assert heat.shape == (7, 7)
    assert heat[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert heat.max() > 0.05  # có ít nhất 1 vùng mà che đi là mạng mất tự tin


def test_saliency_matches_numeric_gradient_and_leaves_model_untouched(trained):
    model, x_val, y_val = trained
    image, label = x_val[1], int(y_val[1])
    before = [{k: v.copy() for k, v in layer.params.items()} for layer in model]
    state = [(layer.__dict__.get("cache"), layer.grads) for layer in model]

    sal = saliency_map(model, image, label)

    # model gốc (dùng chung trong cache) không bị ghi cache/grads mới
    for layer, (cache, grads) in zip(model, state):
        assert layer.__dict__.get("cache") is cache
        assert layer.grads is grads

    def logit(img):
        x = img[None]
        for layer in model:
            x = layer.forward(x, train=False)
        return x[0, label]

    i, j = np.unravel_index(sal.argmax(), sal.shape)
    bumped, dropped = image.copy(), image.copy()
    bumped[i, j, 0] += 1e-5
    dropped[i, j, 0] -= 1e-5
    numeric = abs(logit(bumped) - logit(dropped)) / 2e-5
    assert sal[i, j] == pytest.approx(numeric, rel=1e-3)
    for layer, params in zip(model, before):
        for k, v in params.items():
            np.testing.assert_array_equal(layer.params[k], v)


def test_confusion_matrix_counts():
    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), n_classes=3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 1, 0]])


def test_random_network_representation_is_less_separable_than_trained(trained):
    model, x_val, y_val = trained
    x_tr, y_tr, _, _ = load_digits_split(500)

    def logits(m, x):
        for layer in m:
            x = layer.forward(x, train=False)
        return x

    random_model = build_model(TrainConfig())
    acc_trained = nearest_centroid_accuracy(logits(model, x_tr), y_tr, logits(model, x_val), y_val)
    acc_random = nearest_centroid_accuracy(logits(random_model, x_tr), y_tr, logits(random_model, x_val), y_val)
    assert acc_trained > 0.95 > 0.8 > acc_random


def test_center_concentration_uniform_vs_peaked():
    uniform = np.zeros((20, 20))
    uniform[4:16, 4:16] = 1.0  # receptive field 12×12, ảnh hưởng đều
    share, area, edge = center_concentration(uniform, 12)
    assert share == pytest.approx(area)
    assert edge == pytest.approx(1.0)

    yy, xx = np.mgrid[:20, :20]
    peaked = np.exp(-((yy - 9.5) ** 2 + (xx - 9.5) ** 2) / 8.0) * (uniform > 0)
    share, area, edge = center_concentration(peaked, 12)
    assert share > 2 * area
    assert edge < 0.1
