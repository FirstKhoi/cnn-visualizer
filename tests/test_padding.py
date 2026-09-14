import numpy as np

from cnn_core.padding import pad_matrix


def test_pad_zero_shape_and_values():
    matrix = np.array([[1, 2], [3, 4]])
    result = pad_matrix(matrix, pad=1)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_pad_zero_is_noop():
    matrix = np.array([[5, 6], [7, 8]])
    result = pad_matrix(matrix, pad=0)
    np.testing.assert_array_equal(result, matrix)


def test_pad_two():
    matrix = np.array([[1]])
    result = pad_matrix(matrix, pad=2)
    assert result.shape == (5, 5)
    assert result[2, 2] == 1
    assert result.sum() == 1
