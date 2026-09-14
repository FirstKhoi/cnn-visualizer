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
    raise NotImplementedError("TODO Phase 4: implement max_pool2d trong cnn_core/pooling.py")


def avg_pool2d(input: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Average pooling — giống max_pool2d nhưng dùng np.mean() thay np.max()."""
    raise NotImplementedError("TODO Phase 4: implement avg_pool2d trong cnn_core/pooling.py")
