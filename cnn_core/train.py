"""Train 1 CNN nhỏ trên sklearn digits (8×8, 10 lớp) bằng cnn_core/layers.py.

Mọi thứ numpy thuần: forward -> softmax + cross-entropy -> backward -> SGD
momentum (+ weight decay). Đủ nhanh (vài giây) để trang Streamlit train live
và so sánh các kỹ thuật chống overfit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.datasets import load_digits

from cnn_core.layers import (
    BatchNorm2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Layer,
    MaxPool2D,
    ReLU,
    cross_entropy,
    softmax,
    softmax_cross_entropy_backward,
)

VAL_SIZE = 500
MAX_TRAIN_SIZE = 1797 - VAL_SIZE


@dataclass(frozen=True)
class TrainConfig:
    use_bn: bool = True
    dropout: float = 0.0
    weight_decay: float = 0.0
    augment: bool = False
    label_noise: float = 0.0
    train_size: int = MAX_TRAIN_SIZE
    epochs: int = 15
    lr: float = 0.01
    batch_size: int = 32
    seed: int = 0


@lru_cache(maxsize=1)
def _digits() -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    x = (d.images / 16.0).reshape(-1, 8, 8, 1)
    # hoán vị cố định: val luôn là cùng 500 ảnh dù train_size/seed đổi thế nào
    idx = np.random.default_rng(12345).permutation(len(x))
    return x[idx], d.target[idx]


def load_digits_split(
    train_size: int = MAX_TRAIN_SIZE, label_noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (x_train, y_train, x_val, y_val), ảnh shape (N, 8, 8, 1) trong [0, 1].

    label_noise: tỉ lệ ảnh TRAIN bị đổi sang 1 nhãn SAI ngẫu nhiên (val luôn sạch) —
    mô phỏng dữ liệu gán nhãn sai để thấy mạng "học thuộc" nhiễu ra sao.
    """
    if not 1 <= train_size <= MAX_TRAIN_SIZE:
        raise ValueError(f"train_size phải trong [1, {MAX_TRAIN_SIZE}], nhận {train_size}")
    if not 0.0 <= label_noise <= 1.0:
        raise ValueError(f"label_noise phải trong [0, 1], nhận {label_noise}")
    x, y = _digits()
    x_tr, y_tr = x[VAL_SIZE : VAL_SIZE + train_size], y[VAL_SIZE : VAL_SIZE + train_size].copy()
    if label_noise > 0:
        rng = np.random.default_rng(7)
        flip = rng.random(len(y_tr)) < label_noise
        # cộng thêm 1..9 (mod 10) -> nhãn bị chọn chắc chắn đổi sang lớp KHÁC
        y_tr[flip] = (y_tr[flip] + rng.integers(1, 10, flip.sum())) % 10
    return x_tr, y_tr, x[:VAL_SIZE], y[:VAL_SIZE]


def build_model(cfg: TrainConfig) -> list[Layer]:
    """Conv(16) -> [BN] -> ReLU -> Pool -> Conv(32) -> [BN] -> ReLU -> Pool
    -> Flatten -> Dense(128 -> 64) -> ReLU -> [Dropout] -> Dense(64 -> 10).
    Shape: 8×8×1 -> 4×4×16 -> 2×2×32 -> 128 -> 64 -> 10.

    Cố ý dư sức chứa (~15k tham số cho vài trăm ảnh) để overfit hiện rõ ở trang 12.
    """
    rng = np.random.default_rng(cfg.seed)
    model: list[Layer] = [Conv2D(1, 16, 3, padding=1, rng=rng)]
    if cfg.use_bn:
        model.append(BatchNorm2D(16))
    model += [ReLU(), MaxPool2D(2, 2), Conv2D(16, 32, 3, padding=1, rng=rng)]
    if cfg.use_bn:
        model.append(BatchNorm2D(32))
    model += [ReLU(), MaxPool2D(2, 2), Flatten(), Dense(2 * 2 * 32, 64, rng=rng), ReLU()]
    if cfg.dropout > 0:
        model.append(Dropout(cfg.dropout, rng=rng))
    model.append(Dense(64, 10, rng=rng))
    return model


def forward(model: list[Layer], x: np.ndarray, train: bool = False) -> np.ndarray:
    for layer in model:
        x = layer.forward(x, train=train)
    return x


def forward_trace(model: list[Layer], x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Chạy eval-mode, giữ output của TỪNG lớp: [("Input", x), ("Conv2D", ...), ...]."""
    trace = [("Input", x)]
    for layer in model:
        x = layer.forward(x, train=False)
        trace.append((type(layer).__name__, x))
    return trace


def predict(model: list[Layer], x: np.ndarray) -> np.ndarray:
    """-> xác suất shape (N, 10)."""
    return softmax(forward(model, x, train=False))


def evaluate(model: list[Layer], x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """-> (loss, accuracy) ở eval mode."""
    probs = predict(model, x)
    return cross_entropy(probs, y), float(np.mean(probs.argmax(axis=1) == y))


def augment_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dịch mỗi ảnh ngẫu nhiên -1/0/+1 pixel theo mỗi chiều (viền mới = 0)."""
    N, H, W, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    dy, dx = rng.integers(0, 3, N), rng.integers(0, 3, N)
    return np.stack([padded[n, dy[n] : dy[n] + H, dx[n] : dx[n] + W] for n in range(N)])


def param_layer_names(model: list[Layer]) -> list[str]:
    """Tên các lớp có weight "W" (conv/dense), đánh số theo thứ tự: ["Conv2D 1", "Conv2D 2", "Dense 1"]."""
    names, counts = [], {}
    for layer in model:
        if "W" in layer.params:
            kind = type(layer).__name__
            counts[kind] = counts.get(kind, 0) + 1
            names.append(f"{kind} {counts[kind]}")
    return names


def train(cfg: TrainConfig) -> tuple[list[Layer], dict[str, list]]:
    """-> (model, history). history mỗi key 1 list theo epoch:
    train_loss, train_acc, val_loss, val_acc (đo ở eval mode), và grad_norms
    (list các ||dW|| trung bình theo lớp, thứ tự như param_layer_names)."""
    x_tr, y_tr, x_val, y_val = load_digits_split(cfg.train_size, cfg.label_noise)
    model = build_model(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    velocity = [{k: np.zeros_like(v) for k, v in layer.params.items()} for layer in model]
    weighted = [layer for layer in model if "W" in layer.params]
    history: dict[str, list] = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "grad_norms")}

    for _ in range(cfg.epochs):
        order = rng.permutation(len(x_tr))
        norm_sums = np.zeros(len(weighted))
        n_batches = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            xb, yb = x_tr[batch], y_tr[batch]
            if cfg.augment:
                xb = augment_shift(xb, rng)
            probs = softmax(forward(model, xb, train=True))
            dout = softmax_cross_entropy_backward(probs, yb)
            for layer in reversed(model):
                dout = layer.backward(dout)

            norm_sums += [np.linalg.norm(layer.grads["W"]) for layer in weighted]
            n_batches += 1
            for layer, vel in zip(model, velocity):
                for k, param in layer.params.items():
                    grad = layer.grads[k]
                    if k == "W":  # weight decay chỉ phạt W, không phạt bias / gamma / beta của BN
                        grad = grad + cfg.weight_decay * param
                    vel[k] = 0.9 * vel[k] - cfg.lr * grad
                    param += vel[k]

        for split, (xs, ys) in (("train", (x_tr, y_tr)), ("val", (x_val, y_val))):
            loss, acc = evaluate(model, xs, ys)
            history[f"{split}_loss"].append(loss)
            history[f"{split}_acc"].append(acc)
        history["grad_norms"].append(list(norm_sums / n_batches))

    return model, history
