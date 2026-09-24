"""Phase 5 — Ghép thành ConvBlock.

Một "block" trong CNN thật (LeNet, VGG, ResNet...) gần như luôn là
Conv -> Activation -> Pool lặp lại. Không train ở project này (không có
backprop) — kernel được sinh ngẫu nhiên chỉ để QUAN SÁT forward pass đi qua
nhiều lớp trông ra sao, giống việc nhìn model lúc mới init trước khi train.
"""

import numpy as np

from cnn_core.activation import relu
from cnn_core.convolution import conv2d_multichannel
from cnn_core.pooling import max_pool2d


def make_random_kernels(num_kernels: int, kernel_size: int, in_channels: int, seed: int | None = None) -> np.ndarray:
    """Sinh bộ kernel ngẫu nhiên cho 1 ConvBlock (đã implement sẵn, không phải TODO).

    Returns:
        Mảng shape (kernel_size, kernel_size, in_channels, num_kernels),
        giá trị ~ N(0, 0.1) — nhỏ để feature map không bão hòa ngay.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(kernel_size, kernel_size, in_channels, num_kernels))


class ConvBlock:
    """Conv -> ReLU -> MaxPool, đúng thứ tự chuẩn của 1 block CNN cổ điển."""

    def __init__(
        self,
        kernels: np.ndarray,
        bias: np.ndarray | None = None,
        conv_stride: int = 1,
        conv_padding: int = 0,
        pool_size: int = 2,
        pool_stride: int = 2,
    ):
        """
        Args:
            kernels: shape (K, K, C_in, C_out), vd từ make_random_kernels().
            bias: shape (C_out,) hoặc None.
            conv_stride, conv_padding: tham số cho conv2d_multichannel.
            pool_size, pool_stride: tham số cho max_pool2d, áp dụng riêng
                từng kênh sau activation.

        Hint: chỉ cần lưu lại các tham số vào self.*, không có logic phức
        tạp ở __init__ — toàn bộ tính toán nằm trong forward().
        """
        K, _, C_in, C_out = kernels.shape
        if bias is not None and bias.shape != (C_out,):
            raise ValueError(f"Shape of bias {bias.shape} do not match with C_out ({C_out},)")
        self.kernels = kernels
        self.bias = bias
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def forward_steps(self, x: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Giống forward() nhưng giữ lại kết quả SAU TỪNG BƯỚC, để trang Full
        Pipeline đi qua được Conv -> ReLU -> Pool thay vì chỉ thấy kết quả cuối.

        Returns:
            [("conv", y1), ("relu", y2), ("pool", y3)] — y3 chính là forward(x).
        """
        conv = conv2d_multichannel(x, self.kernels, self.bias, stride=self.conv_stride, padding=self.conv_padding)
        activated = relu(conv)
        H1, W1, C_out = activated.shape
        H_out = (H1 - self.pool_size) // self.pool_stride + 1
        W_out = (W1 - self.pool_size) // self.pool_stride + 1

        pooled = np.zeros((H_out, W_out, C_out), dtype=activated.dtype)

        for c in range(C_out):
            pooled[:, :, c] = max_pool2d(
                activated[:, :, c],
                size=self.pool_size,
                stride=self.pool_stride
            )

        return [("conv", conv), ("relu", activated), ("pool", pooled)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Chạy x qua conv -> relu -> pool, trả về kết quả sau pool.

        Args:
            x: mảng 3D shape (H, W, C_in).

        Returns:
            Mảng 3D shape (H_out, W_out, C_out) sau cả 3 bước.
        """
        return self.forward_steps(x)[-1][1]
