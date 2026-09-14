import numpy as np

from cnn_core.activation import relu, sigmoid


def test_relu_clips_negatives():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(relu(x), expected)


def test_relu_preserves_shape():
    x = np.random.default_rng(0).normal(size=(3, 4))
    assert relu(x).shape == x.shape


def test_sigmoid_bounds_and_midpoint():
    x = np.array([-100.0, 0.0, 100.0])
    result = sigmoid(x)
    assert np.isclose(result[1], 0.5)
    assert result[0] < 0.01
    assert result[2] > 0.99
