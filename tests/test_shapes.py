import pytest

from cnn_core.shapes import conv_output_shape


def test_shape_no_padding_stride1():
    # (5 - 3)/1 + 1 = 3
    assert conv_output_shape(input_size=5, kernel_size=3, stride=1, padding=0) == 3


def test_shape_with_padding():
    # (5 - 3 + 2*1)/1 + 1 = 5  (padding "same")
    assert conv_output_shape(input_size=5, kernel_size=3, stride=1, padding=1) == 5


def test_shape_with_stride():
    # (7 - 3)/2 + 1 = 3
    assert conv_output_shape(input_size=7, kernel_size=3, stride=2, padding=0) == 3


def test_shape_invalid_raises():
    # (6 - 3)/2 = 1.5 -> không chia hết -> phải raise
    with pytest.raises(ValueError):
        conv_output_shape(input_size=6, kernel_size=3, stride=2, padding=0)
