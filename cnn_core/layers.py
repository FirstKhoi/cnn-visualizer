"""Các lớp mạng có forward + backward, vectorized — dùng để TRAIN (trang 9–12).

Khác cnn_core/convolution.py (vòng lặp tay để dạy cơ chế): ở đây conv/pool
dùng "cửa sổ" (im2col) + matmul cho đủ nhanh để train trong vài giây.
tests/test_layers.py đối chiếu 2 bản với nhau.

Quy ước: batch ảnh shape (N, H, W, C); Dense nhận (N, D). Mỗi lớp có
forward(x, train) và backward(dout); lớp có tham số giữ `params` và `grads`
(dict cùng key) để optimizer cập nhật.
"""

from __future__ import annotations

import numpy as np


def _windows(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    """(N, H, W, C) -> (N, H_out, W_out, k, k, C): mọi cửa sổ k×k mà kernel/pool
    sẽ "nhìn", xếp sẵn để tính 1 lần thay vì vòng lặp qua từng vị trí."""
    N, H, W, C = x.shape
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    win = np.empty((N, H_out, W_out, k, k, C), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            win[:, :, :, i, j, :] = x[:, i : i + stride * H_out : stride, j : j + stride * W_out : stride, :]
    return win


def _windows_backward(dwin: np.ndarray, x_shape: tuple, k: int, stride: int) -> np.ndarray:
    """Ngược của _windows: cộng dồn gradient từ mọi cửa sổ về đúng pixel gốc
    (1 pixel nằm trong nhiều cửa sổ -> nhận tổng gradient của tất cả)."""
    _, H_out, W_out = dwin.shape[:3]
    dx = np.zeros(x_shape, dtype=dwin.dtype)
    for i in range(k):
        for j in range(k):
            dx[:, i : i + stride * H_out : stride, j : j + stride * W_out : stride, :] += dwin[:, :, :, i, j, :]
    return dx


class Layer:
    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Conv2D(Layer):
    """Weight shape (K, K, C_in, C_out) — cùng quy ước với conv2d_multichannel."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int = 0, rng=None):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        # He init: phương sai 2/fan_in giữ độ lớn activation ổn định qua ReLU
        std = np.sqrt(2.0 / (k * k * in_ch))
        self.params = {"W": rng.normal(0.0, std, (k, k, in_ch, out_ch)), "b": np.zeros(out_ch)}
        self.k, self.stride, self.padding = k, stride, padding

    def forward(self, x, train=True):
        p = self.padding
        if p:
            x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        win = _windows(x, self.k, self.stride)
        N, H_out, W_out = win.shape[:3]
        cols = win.reshape(N * H_out * W_out, -1)
        W = self.params["W"]
        self.cache = (x.shape, cols, win.shape)
        out = cols @ W.reshape(-1, W.shape[-1]) + self.params["b"]
        return out.reshape(N, H_out, W_out, -1)

    def backward(self, dout):
        x_shape, cols, win_shape = self.cache
        W = self.params["W"]
        d = dout.reshape(-1, W.shape[-1])
        self.grads = {"W": (cols.T @ d).reshape(W.shape), "b": d.sum(axis=0)}
        dwin = (d @ W.reshape(-1, W.shape[-1]).T).reshape(win_shape)
        dx = _windows_backward(dwin, x_shape, self.k, self.stride)
        p = self.padding
        return dx[:, p:-p, p:-p, :] if p else dx


class ReLU(Layer):
    def forward(self, x, train=True):
        # tính vào biến cục bộ: không đọc lại self.mask, để 2 phiên dùng chung 1
        # model đã cache (st.cache_resource) không giẫm lên mask của nhau
        mask = x > 0
        self.mask = mask
        return x * mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D(Layer):
    def __init__(self, size: int = 2, stride: int = 2):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x, train=True):
        win = _windows(x, self.size, self.stride)
        N, H_out, W_out, s, _, C = win.shape
        flat = win.reshape(N, H_out, W_out, s * s, C)
        # gradient chỉ chảy về đúng 1 ô lớn nhất trong mỗi cửa sổ
        self.argmax = flat.argmax(axis=3)
        self.cache = (x.shape, win.shape)
        return flat.max(axis=3)

    def backward(self, dout):
        x_shape, win_shape = self.cache
        N, H_out, W_out, s, _, C = win_shape
        onehot = self.argmax[:, :, :, None, :] == np.arange(s * s)[None, None, None, :, None]
        dwin = (onehot * dout[:, :, :, None, :]).reshape(win_shape)
        return _windows_backward(dwin, x_shape, s, self.stride)


class BatchNorm2D(Layer):
    """Chuẩn hoá từng kênh về mean 0 / std 1 theo mọi trục trừ trục kênh
    (N, H, W với ảnh; N với vector), rồi scale/shift bằng gamma/beta học được.

    train=True: dùng mean/var của CHÍNH batch này + cập nhật running stats.
    train=False: dùng running stats (1 ảnh lẻ lúc predict không có "batch").
    backward() dùng đúng công thức của mode ở lần forward gần nhất.
    """

    def __init__(self, ch: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.params = {"gamma": np.ones(ch), "beta": np.zeros(ch)}
        self.running_mean = np.zeros(ch)
        self.running_var = np.ones(ch)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, train=True):
        axes = tuple(range(x.ndim - 1))
        if train:
            mean, var = x.mean(axis=axes), x.var(axis=axes)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mean
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mean, var = self.running_mean, self.running_var
        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        self.cache = (x_hat, std, axes, train)
        return self.params["gamma"] * x_hat + self.params["beta"]

    def backward(self, dout):
        x_hat, std, axes, train = self.cache
        self.grads = {"gamma": (dout * x_hat).sum(axis=axes), "beta": dout.sum(axis=axes)}
        dx_hat = dout * self.params["gamma"]
        if not train:
            # eval: mean/std là hằng số (running stats) -> BN chỉ là phép affine
            return dx_hat / std
        M = dout.size // dout.shape[-1]  # số phần tử mỗi kênh
        # train: mean/var phụ thuộc cả batch nên gradient có thêm 2 số hạng
        return (M * dx_hat - dx_hat.sum(axis=axes) - x_hat * (dx_hat * x_hat).sum(axis=axes)) / (M * std)


class Dropout(Layer):
    """Inverted dropout: lúc train tắt ngẫu nhiên tỉ lệ p neuron và nhân phần
    còn lại với 1/(1-p) để kỳ vọng không đổi -> lúc eval chỉ cần để nguyên."""

    def __init__(self, p: float, rng=None):
        super().__init__()
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = np.ones_like(x)
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Flatten(Layer):
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense(Layer):
    def __init__(self, in_dim: int, out_dim: int, rng=None):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        self.params = {"W": rng.normal(0.0, np.sqrt(1.0 / in_dim), (in_dim, out_dim)), "b": np.zeros(out_dim)}

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads = {"W": self.x.T @ dout, "b": dout.sum(axis=0)}
        return dout @ self.params["W"].T


def softmax(logits: np.ndarray) -> np.ndarray:
    """Trừ max mỗi hàng trước khi exp — kết quả y hệt nhưng không tràn số."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Loss = trung bình -log(xác suất model gán cho lớp ĐÚNG)."""
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def softmax_cross_entropy_backward(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dLoss/dlogits = (p - onehot(y)) / N — gọn đến bất ngờ nhờ softmax+CE đi cặp."""
    d = probs.copy()
    d[np.arange(len(y)), y] -= 1
    return d / len(y)
