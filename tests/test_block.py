import numpy as np

from cnn_core.block import ConvBlock, make_random_kernels


def test_make_random_kernels_shape():
    kernels = make_random_kernels(num_kernels=4, kernel_size=3, in_channels=3, seed=0)
    assert kernels.shape == (3, 3, 3, 4)


def test_conv_block_forward_deterministic():
    """kernel 2x2 toàn số 1, 1 kênh -> conv giống test_convolution basic,
    relu là no-op (toàn số dương), rồi max_pool size=2 stride=1 trên kết
    quả conv 3x3 -> output 2x2 (tính tay)."""
    input_4x4 = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=float,
    ).reshape(4, 4, 1)

    kernels = np.ones((2, 2, 1, 1))  # 1 kernel, 1 kênh input, toàn số 1
    block = ConvBlock(
        kernels=kernels,
        bias=None,
        conv_stride=1,
        conv_padding=0,
        pool_size=2,
        pool_stride=1,
    )
    result = block.forward(input_4x4)

    # conv output (trước pool) = [[14,18,22],[30,34,38],[46,50,54]]
    expected = np.array([[34, 38], [50, 54]]).reshape(2, 2, 1)
    assert result.shape == (2, 2, 1)
    np.testing.assert_array_equal(result, expected)


def test_conv_block_forward_shape_multichannel():
    rng = np.random.default_rng(0)
    input_rgb = rng.normal(size=(8, 8, 3))
    kernels = make_random_kernels(num_kernels=5, kernel_size=3, in_channels=3, seed=0)
    block = ConvBlock(kernels=kernels, conv_stride=1, conv_padding=0, pool_size=2, pool_stride=2)
    result = block.forward(input_rgb)
    # conv: (8-3)+1=6 -> (6,6,5); pool size2 stride2: (6-2)//2+1=3 -> (3,3,5)
    assert result.shape == (3, 3, 5)
