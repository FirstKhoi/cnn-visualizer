"""Phase 4 — Pooling.

Pooling thu nhỏ feature map bằng cách tóm tắt từng vùng nhỏ thành 1 số —
giảm kích thước + giúp model bớt nhạy với dịch chuyển nhỏ (translation
invariance), khác hẳn mục đích của convolution (trích đặc trưng).
"""

import numpy as np


def max_pool2d(input: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Max pooling trên ma trận 2D.

    Args:
        input: mảng 2D shape (H, W).
        size: kích thước cửa sổ pooling (vuông, size x size).
        stride: bước nhảy của cửa sổ.

    Returns:
        Mảng 2D shape (H_out, W_out), H_out = (H - size)//stride + 1.

    Hint:
        Cấu trúc vòng lặp giống hệt conv2d (Phase 2) — trượt 1 cửa sổ
        size x size qua input — nhưng thay vì nhân với kernel rồi cộng,
        lấy np.max() của cửa sổ đó. Không có padding, không có "kernel"
        chứa trọng số.
    """
    
    H, W = input.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    
    output = np.zeros((H_out, W_out), dtype=input.dtype)
    
    for i in range(H_out):
        for j in range(W_out):
            window = input[i * stride : i * stride + size, j * stride : j * stride + size]
            output[i, j] = np.max(window)
    return output


def avg_pool2d(input: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Average pooling on a 2D matrix."""
    H, W = input.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    
    # Đảm bảo output luôn là kiểu số thực để giữ phần thập phân của np.mean
    out_dtype = np.float32 if np.issubdtype(input.dtype, np.integer) else input.dtype
    output = np.zeros((H_out, W_out), dtype=out_dtype)
    
    for i in range(H_out):
        for j in range(W_out):
            window = input[i * stride : i * stride + size, j * stride : j * stride + size]
            output[i, j] = np.mean(window)
            
    return output
