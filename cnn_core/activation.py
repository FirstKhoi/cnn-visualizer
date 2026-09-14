"""Phase 4 — Activation.

Không có non-linearity, xếp chồng nhiều lớp conv (vốn là phép tuyến tính)
tương đương với đúng 1 lớp conv duy nhất — activation là thứ cho phép mạng
học được pattern phức tạp hơn tuyến tính.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x) áp dụng element-wise.

    Hint: 1 dòng với np.maximum(0, x). Không dùng vòng lặp.
    """
    raise NotImplementedError("TODO Phase 4: implement relu trong cnn_core/activation.py")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + e^-x) áp dụng element-wise.

    Hint: dùng np.exp(). Chỉ để so sánh với ReLU trên page Activation,
    không dùng trong ConvBlock ở Phase 5.
    """
    raise NotImplementedError("TODO Phase 4: implement sigmoid trong cnn_core/activation.py")
