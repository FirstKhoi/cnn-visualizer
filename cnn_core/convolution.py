"""Phase 2 — Convolution.

Convolution 2D = trượt kernel qua input, ở mỗi vị trí nhân-từng-phần-tử rồi
cộng lại thành 1 số. Đây chính xác là cross-correlation (kernel KHÔNG bị lật
180 độ) — quy ước chuẩn trong deep learning, khác "convolution" toán thuần
túy, nhưng cả field đều gọi tắt là "convolution".
"""

import numpy as np

from cnn_core.padding import pad_matrix


def conv2d(input: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Convolution 2D, 1 kênh.

    Args:
        input: mảng 2D shape (H, W).
        kernel: mảng 2D shape (K, K) (giả định kernel vuông).
        stride: bước nhảy của kernel mỗi lần trượt (cả 2 chiều).
        padding: số pad thêm vào input trước khi convolve (dùng pad_matrix).

    Returns:
        Mảng 2D shape (H_out, W_out) với
        H_out = (H + 2*padding - K) // stride + 1 (tương tự W_out).

    Hint:
        1. Gọi pad_matrix(input, padding) trước.
        2. Tính H_out, W_out (sẽ trùng công thức viết lại ở Phase 3
           cnn_core/shapes.py — có thể import conv_output_shape từ đó nếu
           đã implement, hoặc tính trực tiếp ở đây trước).
        3. Vòng lặp for i, j chạy qua từng vị trí output. Với mỗi (i, j):
           - cắt ra "cửa sổ" input tại hàng [i*stride : i*stride+K],
             cột [j*stride : j*stride+K]
           - output[i, j] = tổng(cửa sổ * kernel)  (element-wise rồi sum,
             KHÔNG phải matrix multiply @)
        4. Không dùng np.convolve, scipy.signal.convolve/correlate,
           hay torch — mục tiêu Phase này là tự tay viết vòng lặp trượt.
    """
    raise NotImplementedError("TODO Phase 2: implement conv2d trong cnn_core/convolution.py")


def conv2d_multichannel(
    input: np.ndarray,
    kernels: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Convolution 2D, nhiều kênh input + nhiều kernel (Phase 5).

    Dùng cho ConvBlock trên ảnh thật (vd RGB, 3 kênh) với nhiều filter cùng
    lúc — mỗi filter cho ra đúng 1 feature map.

    Args:
        input: mảng 3D shape (H, W, C_in).
        kernels: mảng 4D shape (K, K, C_in, C_out) — C_out kernel, mỗi kernel
                 có C_in kênh khớp với input.
        bias: mảng 1D shape (C_out,) cộng vào mỗi feature map, hoặc None.
        stride, padding: giống conv2d.

    Returns:
        Mảng 3D shape (H_out, W_out, C_out).

    Hint:
        - Với mỗi kernel thứ c trong C_out kernel: convolve từng kênh input
          với kênh tương ứng của kernel đó (dùng conv2d ở trên cho từng
          cặp kênh), rồi CỘNG kết quả của C_in kênh lại thành 1 feature map
          duy nhất (không phải concat — 1 kernel đa kênh cho ra 1 output
          map, không phải C_in map).
        - Ghép C_out feature map lại theo trục cuối -> shape (H_out, W_out, C_out).
        - Đừng tối ưu sớm (vectorize/einsum) — vòng lặp lồng nhau gọi lại
          conv2d() là đủ và giúp thấy rõ cơ chế; có thể tối ưu sau ở mục
          "Ý mở rộng" trong TODO.md.
    """
    raise NotImplementedError("TODO Phase 5: implement conv2d_multichannel trong cnn_core/convolution.py")
