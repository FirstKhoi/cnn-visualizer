import numpy as np

from cnn_core.convolution import conv2d


INPUT_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
)


def test_conv2d_basic():
    """kernel 2x2 toàn số 1 => mỗi output = tổng 4 số trong cửa sổ (tính tay)."""
    kernel = np.ones((2, 2))
    result = conv2d(INPUT_4x4, kernel, stride=1, padding=0)
    expected = np.array(
        [
            [14, 18, 22],
            [30, 34, 38],
            [46, 50, 54],
        ]
    )
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, expected)


def test_conv2d_stride():
    """Cùng input/kernel ở trên nhưng stride=2 => chỉ giữ lại 4 vị trí góc."""
    kernel = np.ones((2, 2))
    result = conv2d(INPUT_4x4, kernel, stride=2, padding=0)
    expected = np.array([[14, 22], [46, 54]])
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, expected)


def test_conv2d_padding():
    """kernel 1x1 giá trị 2 => conv chỉ là nhân vô hướng; padding=1 bọc viền 0
    quanh input 2x2 trước, nên output 4x4 với viền = 0."""
    input_2x2 = np.array([[1, 2], [3, 4]])
    kernel = np.array([[2]])
    result = conv2d(input_2x2, kernel, stride=1, padding=1)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 4, 0],
            [0, 6, 8, 0],
            [0, 0, 0, 0],
        ]
    )
    assert result.shape == (4, 4)
    np.testing.assert_array_equal(result, expected)


def test_conv2d_output_shape_formula():
    """H_out phải khớp (H + 2P - K) // S + 1 cho vài bộ tham số ngẫu nhiên."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = rng.integers(5, 12)
        k = rng.integers(1, 4)
        s = rng.integers(1, 3)
        p = rng.integers(0, 3)
        if (h + 2 * p - k) < 0 or (h + 2 * p - k) % s != 0:
            continue
        input = rng.normal(size=(h, h))
        kernel = rng.normal(size=(k, k))
        result = conv2d(input, kernel, stride=int(s), padding=int(p))
        expected_size = (h + 2 * p - k) // s + 1
        assert result.shape == (expected_size, expected_size)
