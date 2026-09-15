"""Phase 1 — Padding.

Thêm viền quanh ma trận 2D trước khi convolution, để (a) output không bị co
lại quá nhanh qua nhiều lớp, và (b) pixel ở rìa ảnh cũng được kernel "nhìn"
đủ số lần như pixel ở giữa.
"""

import numpy as np


def pad_matrix(matrix: np.ndarray, pad: int, mode: str = "zero") -> np.ndarray:
    """Thêm viền dày `pad` quanh `matrix` (2D).

    Args:
        matrix: mảng 2D shape (H, W).
        pad: số hàng/cột thêm vào MỖI cạnh (trên, dưới, trái, phải).
             pad=0 phải trả về matrix không đổi (có thể trả bản copy).
        mode: "zero" -> viền toàn số 0.
              (tùy chọn mở rộng ở TODO.md: "reflect" -> phản chiếu giá trị
              gần rìa thay vì 0 — không bắt buộc để pass test).

    Returns:
        Mảng 2D shape (H + 2*pad, W + 2*pad).

    Hint:
        - Đừng dùng np.pad() trực tiếp cho mode="zero" — tự tạo mảng 0 với
          np.zeros(shape_moi) rồi "dán" matrix gốc vào giữa bằng slicing.
          Mục tiêu là hiểu cơ chế, không phải gọi thư viện có sẵn.
        - Ví dụ pad=1 trên ma trận 2x2:
            [[1, 2],        [[0, 0, 0, 0],
             [3, 4]]   -->   [0, 1, 2, 0],
                              [0, 3, 4, 0],
                              [0, 0, 0, 0]]
    """
    if pad < 0:
        raise ValueError("Padding size 'pad' must be a non-negative integer")
    if pad == 0:
        return matrix.copy()
    
    H, W = matrix.shape
    
    if mode == "zero":
        padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=matrix.dtype)
        padded[pad : pad + H, pad : pad + W] = matrix
        return padded
    elif mode == "reflect":
        padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=matrix.dtype)
        padded[pad : pad + H, pad : pad + W] = matrix
        for i in range(pad):
            padded[pad - 1 - i, pad : pad + W] = matrix[i + 1, :]
            padded[pad + H + i, pad : pad + W] = matrix[H - 2 - i, :]
            
        for j in range(pad):
            padded[:, pad - 1 - j] = padded[:, pad + 1 + j]
            padded[:, pad + W + j] = padded[:, pad + W - 2 - j]
            
        return padded
    else:
        raise ValueError(f"Unsupported padding mode: '{mode}'. Supported modes are 'zero' and 'reflect'.")