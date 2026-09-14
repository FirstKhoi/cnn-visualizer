import numpy as np

from cnn_core.pooling import avg_pool2d, max_pool2d


INPUT_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
)


def test_max_pool2d():
    result = max_pool2d(INPUT_4x4, size=2, stride=2)
    expected = np.array([[6, 8], [14, 16]])
    np.testing.assert_array_equal(result, expected)


def test_avg_pool2d():
    result = avg_pool2d(INPUT_4x4, size=2, stride=2)
    expected = np.array([[3.5, 5.5], [11.5, 13.5]])
    np.testing.assert_allclose(result, expected)


def test_max_pool2d_overlapping_stride():
    result = max_pool2d(INPUT_4x4, size=2, stride=1)
    assert result.shape == (3, 3)
    # góc trên trái: cửa sổ {1,2,5,6} -> max 6
    assert result[0, 0] == 6
    # góc dưới phải: cửa sổ {11,12,15,16} -> max 16
    assert result[2, 2] == 16
