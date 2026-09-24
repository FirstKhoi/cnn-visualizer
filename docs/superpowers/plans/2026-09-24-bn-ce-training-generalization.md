# BatchNorm, Softmax/CE, Training, Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a numpy training stack (layers with backward, digits dataset, SGD) and 4 new Streamlit pages (9 BatchNorm, 10 Softmax + CE, 11 Backprop & Training, 12 Generalization). Also improve pages 7–8 so activation magnitude is honestly visible.

**Architecture:** `cnn_core/layers.py` holds vectorized layers with `forward(x, train)` / `backward(dout)` on NHWC batches. `cnn_core/train.py` loads sklearn digits, builds the model and trains with SGD momentum + weight decay. Pages read results through a shared `st.cache_resource` wrapper (`viz/widgets.py::cached_train`). New matplotlib helpers in `viz/matrix_view.py` keep the existing palette.

**Tech Stack:** Python, numpy, Streamlit, matplotlib, pandas, scikit-learn (dataset only), pytest. torch is optional, only for one cross-check test.

**Spec:** `docs/superpowers/specs/2026-09-24-bn-ce-training-generalization-design.md`

**Provenance:** Every code block below was prototyped and run in a scratch copy of the repo. Results: 56/56 tests pass, all 13 pages render under `AppTest`, full-data training takes ~1 s with val acc ~0.99. Copy the blocks verbatim.

## Global Constraints

- App math is numpy only. torch appears only in `tests/test_layers.py::test_conv_and_batchnorm_match_torch` via `pytest.importorskip`.
- Leave the existing hand-written loop code in `cnn_core/` (padding, convolution, pooling, activation, shapes) unchanged. The new code goes in `layers.py` / `train.py`.
- Batch layout is `(N, H, W, C)`. Conv weights are `(K, K, C_in, C_out)`, the same convention as `conv2d_multichannel`.
- Dataset: `sklearn.datasets.load_digits`. The 500 val images are fixed. Train holds at most 1297 images (`MAX_TRAIN_SIZE`).
- `TrainConfig` defaults: `use_bn=True, dropout=0.0, weight_decay=0.0, augment=False, label_noise=0.0, train_size=1297, epochs=15, lr=0.01, batch_size=32, seed=0`.
- UI copy is Vietnamese, matching existing pages. Use the palette in `viz/theme.py` and the `hero`/`callout` components.
- Only use Streamlit APIs that exist in the pinned `streamlit==1.38.0`. The environment runs 1.61, so `use_container_width` deprecation warnings are expected and harmless.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Diverging learning rate (page 11, lr = 0.3/0.1 without BN).** Loss becomes NaN/inf. Expected: the page shows an `st.error` explaining that lr is too large, and it does not crash. Pinned by the Task 7 interaction test.
2. **Switching presets on page 12.** Every control must update to the preset values, and "Chạy" must append a run without duplicating an existing one. "Xoá hết" must show the info message, not an error. Pinned by the Task 8 interaction test.
3. **Single image through BN in eval mode.** Prediction on 1 image must not depend on batch-mates. Pinned by `test_batchnorm_eval_uses_running_stats`, with the page 9 metric showing 0.000.
4. **Shared colour scale with negative values.** Pages 8 and 11 must keep 0 at the centre of the colormap. Pinned by the symmetric-scale branch in `fig_feature_maps`, which Task 3 checks.
5. **Out-of-range dataset arguments.** `train_size` 0 or above 1297, or `label_noise` outside [0, 1], must raise `ValueError` with a message, never silently slice. Pinned by `test_load_digits_split_rejects_bad_train_size` in Task 2.

---

### Task 1: Vectorized layers with backward (`cnn_core/layers.py`)

**Files:**
- Create: `cnn_core/layers.py`
- Create: `tests/test_layers.py`
- Modify: `requirements.txt` (add `scikit-learn>=1.3`, needed from Task 2 on; folded in here so the dependency lands first)

**Interfaces:**
- Consumes: `cnn_core.convolution.conv2d_multichannel(input, kernels, bias, stride, padding)` (test only).
- Produces: `Layer` (attrs `params: dict`, `grads: dict`, methods `forward(x, train=True)`, `backward(dout)`), `Conv2D(in_ch, out_ch, k, stride=1, padding=0, rng=None)`, `ReLU()`, `MaxPool2D(size=2, stride=2)`, `BatchNorm2D(ch, momentum=0.9, eps=1e-5)` (attrs `running_mean`, `running_var`, `momentum`, `eps`), `Dropout(p, rng=None)`, `Flatten()`, `Dense(in_dim, out_dim, rng=None)`, `softmax(logits) -> ndarray`, `cross_entropy(probs, y) -> float`, `softmax_cross_entropy_backward(probs, y) -> ndarray`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_layers.py`:

```python
import numpy as np
import pytest

from cnn_core.convolution import conv2d_multichannel
from cnn_core.layers import (
    BatchNorm2D,
    Conv2D,
    Dense,
    Dropout,
    MaxPool2D,
    ReLU,
    cross_entropy,
    softmax,
    softmax_cross_entropy_backward,
)


def numeric_grad(f, x, eps=1e-6):
    """Đạo hàm số bằng sai phân trung tâm: (f(x+eps) - f(x-eps)) / 2eps, từng phần tử."""
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def check_layer_grads(layer, x, rng):
    """So backward() với đạo hàm số của L = sum(forward(x) * g), g random."""
    g = rng.normal(size=layer.forward(x, train=True).shape)
    layer.forward(x, train=True)
    dx = layer.backward(g)

    def loss():
        return float(np.sum(layer.forward(x, train=True) * g))

    np.testing.assert_allclose(dx, numeric_grad(loss, x), atol=1e-6)
    for key, param in layer.params.items():
        np.testing.assert_allclose(layer.grads[key], numeric_grad(loss, param), atol=1e-6)


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv2d_gradients(stride, padding):
    rng = np.random.default_rng(0)
    check_layer_grads(Conv2D(2, 3, 3, stride=stride, padding=padding, rng=rng), rng.normal(size=(2, 5, 5, 2)), rng)


def test_conv2d_matches_handwritten_loop_version():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 7, 3))
    layer = Conv2D(3, 4, 3, stride=2, padding=1, rng=rng)
    layer.params["b"] = rng.normal(size=4)
    expected = conv2d_multichannel(x, layer.params["W"], layer.params["b"], stride=2, padding=1)
    np.testing.assert_allclose(layer.forward(x[None])[0], expected, atol=1e-12)


def test_maxpool_forward_and_gradients():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 4, 4, 3))
    pool = MaxPool2D(2, 2)
    out = pool.forward(x)
    assert out.shape == (2, 2, 2, 3)
    assert out[0, 0, 0, 0] == x[0, :2, :2, 0].max()
    check_layer_grads(MaxPool2D(2, 2), x, rng)


def test_relu_and_dense_gradients():
    rng = np.random.default_rng(3)
    check_layer_grads(ReLU(), rng.normal(size=(3, 5)) + 0.01, rng)
    check_layer_grads(Dense(5, 4, rng=rng), rng.normal(size=(3, 5)), rng)


@pytest.mark.parametrize("shape", [(4, 3, 3, 3), (6, 3)])
def test_batchnorm_gradients(shape):
    rng = np.random.default_rng(4)
    bn = BatchNorm2D(3)
    bn.params["gamma"] = rng.normal(size=3)
    bn.params["beta"] = rng.normal(size=3)
    check_layer_grads(bn, rng.normal(size=shape), rng)


def test_batchnorm_train_normalizes_each_channel():
    rng = np.random.default_rng(5)
    x = rng.normal(loc=[5.0, -3.0], scale=[2.0, 0.1], size=(16, 4, 4, 2))
    out = BatchNorm2D(2).forward(x, train=True)
    np.testing.assert_allclose(out.mean(axis=(0, 1, 2)), 0, atol=1e-9)
    np.testing.assert_allclose(out.std(axis=(0, 1, 2)), 1, atol=1e-3)


def test_batchnorm_eval_uses_running_stats():
    bn = BatchNorm2D(1, momentum=0.0)  # momentum 0 -> running stats = stats của batch cuối
    batch = np.array([[1.0], [3.0]])
    bn.forward(batch, train=True)
    np.testing.assert_allclose(bn.running_mean, [2.0])
    np.testing.assert_allclose(bn.running_var, [1.0])
    single = bn.forward(np.array([[4.0]]), train=False)
    np.testing.assert_allclose(single, [[(4.0 - 2.0) / np.sqrt(1.0 + bn.eps)]])


def test_dropout_train_vs_eval():
    x = np.ones((1000, 10))
    drop = Dropout(0.5, rng=np.random.default_rng(6))
    np.testing.assert_array_equal(drop.forward(x, train=False), x)
    out = drop.forward(x, train=True)
    assert set(np.unique(out)) == {0.0, 2.0}  # inverted dropout: phần còn lại nhân 1/(1-p)
    assert abs(out.mean() - 1.0) < 0.05


def test_softmax_cross_entropy():
    logits = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    probs = softmax(logits)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    y = np.array([0, 2])
    expected = -(np.log(probs[0, 0]) + np.log(probs[1, 2])) / 2
    assert cross_entropy(probs, y) == pytest.approx(expected)


def test_softmax_cross_entropy_gradient():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(4, 5))
    y = np.array([0, 3, 1, 4])
    analytic = softmax_cross_entropy_backward(softmax(logits), y)
    np.testing.assert_allclose(analytic, numeric_grad(lambda: cross_entropy(softmax(logits), y), logits), atol=1e-6)


def test_conv_and_batchnorm_match_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 6, 6, 3))
    conv = Conv2D(3, 4, 3, stride=1, padding=1, rng=rng)
    ours = conv.forward(x)
    theirs = torch.nn.functional.conv2d(
        torch.tensor(x).permute(0, 3, 1, 2),
        torch.tensor(conv.params["W"]).permute(3, 2, 0, 1),
        torch.tensor(conv.params["b"]),
        padding=1,
    ).permute(0, 2, 3, 1)
    np.testing.assert_allclose(ours, theirs.numpy(), atol=1e-10)

    bn_ours = BatchNorm2D(4).forward(ours, train=True)
    bn_theirs = torch.nn.functional.batch_norm(
        torch.tensor(ours).permute(0, 3, 1, 2), None, None, training=True, eps=1e-5
    ).permute(0, 2, 3, 1)
    np.testing.assert_allclose(bn_ours, bn_theirs.numpy(), atol=1e-8)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_layers.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'cnn_core.layers'`

- [ ] **Step 3: Implement** — create `cnn_core/layers.py`:

```python
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
        self.mask = x > 0
        return x * self.mask

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
        self.cache = (x_hat, std, axes)
        return self.params["gamma"] * x_hat + self.params["beta"]

    def backward(self, dout):
        x_hat, std, axes = self.cache
        M = dout.size // dout.shape[-1]  # số phần tử mỗi kênh
        self.grads = {"gamma": (dout * x_hat).sum(axis=axes), "beta": dout.sum(axis=axes)}
        dx_hat = dout * self.params["gamma"]
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
```

- [ ] **Step 4: Add dependency** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/requirements.txt b/requirements.txt
index f0af767..298eb83 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -5,2 +5,3 @@ matplotlib==3.9.2
 pillow==10.4.0
 pytest==8.3.3
+scikit-learn>=1.3
```

Run: `pip install "scikit-learn>=1.3"` (skip if already installed).

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_layers.py -q`
Expected: `14 passed`. The torch test shows as skipped if torch is missing, which is fine.

- [ ] **Step 6: Commit**

```bash
git add cnn_core/layers.py tests/test_layers.py requirements.txt
git commit -m "Add vectorized layers with backward pass for training

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Training loop on digits (`cnn_core/train.py`)

**Files:**
- Create: `cnn_core/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces: `VAL_SIZE = 500`, `MAX_TRAIN_SIZE = 1297`, `TrainConfig` (frozen dataclass, fields listed in Global Constraints), `load_digits_split(train_size=1297, label_noise=0.0) -> (x_tr, y_tr, x_val, y_val)`, `build_model(cfg) -> list[Layer]`, `forward(model, x, train=False)`, `forward_trace(model, x) -> list[tuple[str, ndarray]]` (first entry `("Input", x)`, then `(type(layer).__name__, out)`), `predict(model, x) -> probs (N, 10)`, `evaluate(model, x, y) -> (loss, acc)`, `augment_shift(x, rng)`, `param_layer_names(model) -> ["Conv2D 1", "Conv2D 2", "Dense 1", "Dense 2"]`, `train(cfg) -> (model, history)`. `history` keys are `train_loss, train_acc, val_loss, val_acc, grad_norms`, each a list per epoch. `grad_norms[e]` is a list of 4 floats.

- [ ] **Step 1: Write the failing tests** — create `tests/test_train.py`:

```python
import numpy as np
import pytest

from cnn_core.train import (
    MAX_TRAIN_SIZE,
    VAL_SIZE,
    TrainConfig,
    augment_shift,
    build_model,
    forward_trace,
    load_digits_split,
    param_layer_names,
    train,
)


def test_load_digits_split_shapes_and_fixed_val():
    x_tr, y_tr, x_val, y_val = load_digits_split(100)
    assert x_tr.shape == (100, 8, 8, 1) and y_tr.shape == (100,)
    assert x_val.shape == (VAL_SIZE, 8, 8, 1)
    assert 0.0 <= x_tr.min() and x_tr.max() <= 1.0
    _, _, x_val_2, _ = load_digits_split(MAX_TRAIN_SIZE)
    np.testing.assert_array_equal(x_val, x_val_2)  # val không đổi theo train_size


def test_label_noise_only_touches_train_labels():
    _, y_clean, _, y_val_clean = load_digits_split(500)
    _, y_noisy, _, y_val_noisy = load_digits_split(500, label_noise=0.5)
    changed = np.mean(y_clean != y_noisy)
    assert 0.3 < changed < 0.6  # 50% bị gán lại, ~1/10 trong số đó trùng nhãn cũ
    np.testing.assert_array_equal(y_val_clean, y_val_noisy)


@pytest.mark.parametrize("bad", [0, MAX_TRAIN_SIZE + 1])
def test_load_digits_split_rejects_bad_train_size(bad):
    with pytest.raises(ValueError):
        load_digits_split(bad)


def test_augment_shift_moves_pixels_by_at_most_one():
    x = np.zeros((50, 8, 8, 1))
    x[:, 4, 4, 0] = 1.0
    out = augment_shift(x, np.random.default_rng(0))
    for img in out[..., 0]:
        r, c = np.argwhere(img == 1.0)[0]
        assert abs(r - 4) <= 1 and abs(c - 4) <= 1


def test_forward_trace_shapes():
    model = build_model(TrainConfig(use_bn=True, dropout=0.5))
    trace = forward_trace(model, np.zeros((2, 8, 8, 1)))
    shapes = {name: out.shape for name, out in trace}
    assert trace[0][0] == "Input"
    assert shapes["MaxPool2D"] == (2, 2, 2, 32)  # dict giữ lần xuất hiện cuối
    assert shapes["Flatten"] == (2, 128)
    assert trace[-1][1].shape == (2, 10)
    assert param_layer_names(model) == ["Conv2D 1", "Conv2D 2", "Dense 1", "Dense 2"]


def test_train_default_config_learns_digits():
    _, history = train(TrainConfig(epochs=5))
    assert history["val_acc"][-1] > 0.9
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert len(history["grad_norms"]) == 5 and len(history["grad_norms"][0]) == 4


def test_train_is_deterministic_for_same_seed():
    cfg = TrainConfig(train_size=100, epochs=2, dropout=0.3, augment=True)
    _, h1 = train(cfg)
    _, h2 = train(cfg)
    assert h1["val_loss"] == h2["val_loss"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'cnn_core.train'`

- [ ] **Step 3: Implement** — create `cnn_core/train.py`:

```python
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

    label_noise: tỉ lệ ảnh TRAIN bị gán lại nhãn ngẫu nhiên (val luôn sạch) —
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
        y_tr[flip] = rng.integers(0, 10, flip.sum())
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train.py -q`
Expected: `8 passed`, in roughly 5 s.

- [ ] **Step 5: Commit**

```bash
git add cnn_core/train.py tests/test_train.py
git commit -m "Add numpy training loop on sklearn digits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Shared plumbing — viz helpers, `ConvBlock.forward_steps`, page colours, `cached_train`

**Files:**
- Modify: `viz/matrix_view.py` (`fig_feature_maps` gains `shared_scale`; add `fig_activation_hist`, `RUN_COLORS`, `fig_lines`, `fig_curves`, `fig_bars`)
- Modify: `cnn_core/block.py` (add `forward_steps`; `forward` delegates to it)
- Modify: `viz/theme.py` (`PAGE_COLORS` 9–12)
- Modify: `viz/widgets.py` (add `cached_train`)
- Modify: `tests/test_block.py`

**Interfaces:**
- Consumes: `TrainConfig`, `train`, `Layer` from Tasks 1–2.
- Produces: `fig_feature_maps(feature_maps, titles=None, max_cols=4, cmap=WARM, shared_scale=False)`, `fig_activation_hist(named_arrays, color=PRIMARY, max_cols=4)`, `fig_lines(series: dict[str, list], xlabels=None, xlabel="", ylabel="", logy=False, figsize=(6, 3.4))`, `fig_curves(runs: list[(label, history)], metric="acc"|"loss", figsize=(6, 3.6))`, `fig_bars(values, labels=None, highlight=None, title=None, color=PRIMARY, highlight_color=ACCENT, figsize=(4.2, 2.8))`, `ConvBlock.forward_steps(x) -> [("conv", y1), ("relu", y2), ("pool", y3)]`, `PAGE_COLORS[9..12]`, `cached_train(cfg) -> (model, history)`.

- [ ] **Step 1: Write the failing test** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/tests/test_block.py b/tests/test_block.py
index d4549f7..3c2c06e 100644
--- a/tests/test_block.py
+++ b/tests/test_block.py
@@ -48,2 +48,13 @@ def test_conv_block_forward_shape_multichannel():
     # conv: (8-3)+1=6 -> (6,6,5); pool size2 stride2: (6-2)//2+1=3 -> (3,3,5)
     assert result.shape == (3, 3, 5)
+
+
+def test_conv_block_forward_steps_matches_forward():
+    rng = np.random.default_rng(1)
+    x = rng.normal(size=(8, 8, 3))
+    block = ConvBlock(kernels=make_random_kernels(4, 3, 3, seed=0))
+    steps = block.forward_steps(x)
+    assert [name for name, _ in steps] == ["conv", "relu", "pool"]
+    assert steps[0][1].shape == (6, 6, 4)
+    assert steps[1][1].min() >= 0
+    np.testing.assert_array_equal(steps[-1][1], block.forward(x))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_block.py -q`
Expected: `AttributeError: 'ConvBlock' object has no attribute 'forward_steps'`

- [ ] **Step 3: Implement `forward_steps`** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/cnn_core/block.py b/cnn_core/block.py
index f53bf93..39df60e 100644
--- a/cnn_core/block.py
+++ b/cnn_core/block.py
@@ -58,35 +58,36 @@ class ConvBlock:
         self.pool_stride = pool_stride
 
-    def forward(self, x: np.ndarray) -> np.ndarray:
-        """Chạy x qua conv -> relu -> pool.
-
-        Args:
-            x: mảng 3D shape (H, W, C_in).
+    def forward_steps(self, x: np.ndarray) -> list[tuple[str, np.ndarray]]:
+        """Giống forward() nhưng giữ lại kết quả SAU TỪNG BƯỚC, để trang Full
+        Pipeline đi qua được Conv -> ReLU -> Pool thay vì chỉ thấy kết quả cuối.
 
         Returns:
-            Mảng 3D shape (H_out, W_out, C_out) sau cả 3 bước.
-
-        Hint:
-            1. y = conv2d_multichannel(x, self.kernels, self.bias, stride=self.conv_stride, padding=self.conv_padding)
-               -> shape (H1, W1, C_out)
-            2. y = relu(y)  (áp dụng trên cả mảng, element-wise là đủ)
-            3. max_pool2d chỉ nhận input 2D — áp dụng RIÊNG cho từng kênh
-               (vòng lặp qua C_out, pool từng lát y[:, :, c], rồi ghép lại
-               theo trục cuối) -> shape (H_out, W_out, C_out)
+            [("conv", y1), ("relu", y2), ("pool", y3)] — y3 chính là forward(x).
         """
-        y = conv2d_multichannel(x, self.kernels, self.bias, stride=self.conv_stride, padding=self.conv_padding)
-        y = relu(y)
-        H1, W1, C_out = y.shape
+        conv = conv2d_multichannel(x, self.kernels, self.bias, stride=self.conv_stride, padding=self.conv_padding)
+        activated = relu(conv)
+        H1, W1, C_out = activated.shape
         H_out = (H1 - self.pool_size) // self.pool_stride + 1
         W_out = (W1 - self.pool_size) // self.pool_stride + 1
-        
-        output = np.zeros((H_out, W_out, C_out), dtype=y.dtype)
-        
+
+        pooled = np.zeros((H_out, W_out, C_out), dtype=activated.dtype)
+
         for c in range(C_out):
-            output[:, :, c] = max_pool2d(
-                y[:, :, c],
+            pooled[:, :, c] = max_pool2d(
+                activated[:, :, c],
                 size=self.pool_size,
                 stride=self.pool_stride
             )
-            
-        return output
+
+        return [("conv", conv), ("relu", activated), ("pool", pooled)]
+
+    def forward(self, x: np.ndarray) -> np.ndarray:
+        """Chạy x qua conv -> relu -> pool, trả về kết quả sau pool.
+
+        Args:
+            x: mảng 3D shape (H, W, C_in).
+
+        Returns:
+            Mảng 3D shape (H_out, W_out, C_out) sau cả 3 bước.
+        """
+        return self.forward_steps(x)[-1][1]
```

- [ ] **Step 4: Add viz helpers** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/viz/matrix_view.py b/viz/matrix_view.py
index e2c5790..bcf5455 100644
--- a/viz/matrix_view.py
+++ b/viz/matrix_view.py
@@ -148,4 +148,5 @@ def fig_feature_maps(
     max_cols: int = 4,
     cmap=WARM,
+    shared_scale: bool = False,
 ) -> plt.Figure:
     """Vẽ lưới các feature map từ 1 tensor shape (H, W, C) — mỗi kênh 1 ô.
@@ -153,4 +154,10 @@ def fig_feature_maps(
     Dùng ở trang Conv Block / Full Pipeline để xem tất cả feature map ra từ
     1 lớp conv cùng lúc.
+
+    shared_scale=False: mỗi map tự co giãn màu riêng -> dễ nhìn HÌNH DẠNG
+        nhưng che mất ĐỘ LỚN (map toàn số ~0.001 trông sáng y như map ~10).
+    shared_scale=True: mọi map chung 1 thang màu + 1 colorbar -> so được độ
+        lớn giữa các kênh (thấy kênh nào "chết", kênh nào trội). Nếu có số
+        âm, thang đối xứng quanh 0.
     """
     feature_maps = np.asarray(feature_maps)
@@ -158,9 +165,15 @@ def fig_feature_maps(
     cols = min(max_cols, channels)
     rows = int(np.ceil(channels / cols))
+    vmin = vmax = None
+    if shared_scale:
+        vmin, vmax = float(feature_maps.min()), float(feature_maps.max())
+        if vmin < 0:  # có số âm -> thang đối xứng quanh 0 để 0 luôn ở giữa colormap
+            vmax = max(-vmin, vmax)
+            vmin = -vmax
     fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
     axes = np.atleast_1d(axes).ravel()
     for c in range(channels):
         ax = axes[c]
-        ax.imshow(feature_maps[:, :, c], cmap=cmap)
+        im = ax.imshow(feature_maps[:, :, c], cmap=cmap, vmin=vmin, vmax=vmax)
         ax.set_title(titles[c] if titles else f"kernel {c}", fontsize=10)
         ax.axis("off")
@@ -168,3 +181,118 @@ def fig_feature_maps(
         axes[c].axis("off")
     fig.tight_layout()
+    if shared_scale:
+        fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
+    return fig
+
+
+def fig_activation_hist(
+    named_arrays: list[tuple[str, np.ndarray]],
+    color: str = PRIMARY,
+    max_cols: int = 4,
+) -> plt.Figure:
+    """Histogram giá trị activation cho từng lớp, CHUNG trục x để thấy phân
+    phối co lại / phình ra qua độ sâu. Tiêu đề mỗi ô ghi std và % số 0."""
+    n = len(named_arrays)
+    cols = min(max_cols, n)
+    rows = int(np.ceil(n / cols))
+    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.3 * rows), sharex=True, squeeze=False)
+    axes = axes.ravel()
+    for ax, (name, arr) in zip(axes, named_arrays):
+        values = np.asarray(arr, dtype=float).ravel()
+        ax.hist(values, bins=40, color=color, alpha=0.85)
+        ax.set_title(f"{name}\nstd {values.std():.3g} · zero {100 * np.mean(values == 0):.0f}%", fontsize=9)
+        ax.set_yticks([])
+        for side in ("top", "right", "left"):
+            ax.spines[side].set_visible(False)
+    for ax in axes[n:]:
+        ax.axis("off")
+    fig.tight_layout()
+    return fig
+
+
+RUN_COLORS = ["#6C5CE7", "#E17055", "#00B894", "#4D96FF", "#FD79A8", "#FDCB6E", "#636E72", "#00CEC9"]
+
+
+def fig_lines(
+    series: dict[str, list[float]],
+    xlabels: list[str] | None = None,
+    xlabel: str = "",
+    ylabel: str = "",
+    logy: bool = False,
+    figsize: tuple[float, float] = (6, 3.4),
+) -> plt.Figure:
+    """Nhiều đường trên 1 trục, mỗi series 1 màu. xlabels (tuỳ chọn) đặt tên
+    cho từng điểm trên trục x (vd tên lớp) thay vì số 0, 1, 2..."""
+    fig, ax = plt.subplots(figsize=figsize)
+    for color, (name, values) in zip(RUN_COLORS, series.items()):
+        ax.plot(range(len(values)), values, marker="o", markersize=4, linewidth=2, color=color, label=name)
+    if xlabels is not None:
+        ax.set_xticks(range(len(xlabels)))
+        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
+    if logy:
+        ax.set_yscale("log")
+    ax.set_xlabel(xlabel)
+    ax.set_ylabel(ylabel)
+    ax.grid(alpha=0.25)
+    for side in ("top", "right"):
+        ax.spines[side].set_visible(False)
+    if len(series) > 1:
+        ax.legend(frameon=False, fontsize=8)
+    fig.tight_layout()
+    return fig
+
+
+def fig_curves(runs: list[tuple[str, dict]], metric: str = "acc", figsize: tuple[float, float] = (6, 3.6)) -> plt.Figure:
+    """Đường học train (nét đứt) vs val (nét liền) cho nhiều lần chạy chồng
+    lên nhau. Chấm tròn = epoch có val tốt nhất (nơi early stopping sẽ dừng).
+
+    runs: [(nhãn, history)] với history từ cnn_core.train.train().
+    metric: "acc" hoặc "loss".
+    """
+    fig, ax = plt.subplots(figsize=figsize)
+    for color, (label, hist) in zip(RUN_COLORS, runs):
+        train_vals, val_vals = hist[f"train_{metric}"], hist[f"val_{metric}"]
+        epochs = np.arange(1, len(val_vals) + 1)
+        ax.plot(epochs, train_vals, linestyle="--", linewidth=1.5, color=color, alpha=0.7)
+        ax.plot(epochs, val_vals, linewidth=2.2, color=color, label=label)
+        best = int(np.argmax(val_vals) if metric == "acc" else np.argmin(val_vals))
+        ax.scatter([epochs[best]], [val_vals[best]], color=color, s=40, zorder=3, edgecolor="white")
+    ax.set_xlabel("epoch")
+    ax.set_ylabel("accuracy" if metric == "acc" else "loss (cross-entropy)")
+    ax.set_title("nét đứt = train · nét liền = val · chấm = val tốt nhất", fontsize=9, color=MUTED, fontweight="normal")
+    ax.grid(alpha=0.25)
+    for side in ("top", "right"):
+        ax.spines[side].set_visible(False)
+    ax.legend(frameon=False, fontsize=8)
+    fig.tight_layout()
+    return fig
+
+
+def fig_bars(
+    values: np.ndarray,
+    labels: list[str] | None = None,
+    highlight: int | None = None,
+    title: str | None = None,
+    color: str = PRIMARY,
+    highlight_color: str = ACCENT,
+    figsize: tuple[float, float] = (4.2, 2.8),
+) -> plt.Figure:
+    """Bar chart 1 vector (logits, xác suất, gradient...). highlight = index
+    cột tô màu khác (vd lớp đúng). Có đường 0 để thấy rõ giá trị âm."""
+    values = np.asarray(values, dtype=float)
+    labels = labels if labels is not None else [str(i) for i in range(len(values))]
+    colors = [highlight_color if i == highlight else color for i in range(len(values))]
+    fig, ax = plt.subplots(figsize=figsize)
+    ax.bar(labels, values, color=colors)
+    ax.axhline(0, color=BORDER, linewidth=1)
+    if len(values) <= 12:
+        for i, v in enumerate(values):
+            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=TEXT)
+    else:
+        ax.set_xticks([])
+    for side in ("top", "right"):
+        ax.spines[side].set_visible(False)
+    if title:
+        ax.set_title(title, fontsize=10)
+    fig.tight_layout()
     return fig
```

- [ ] **Step 5: Page colours and `cached_train`** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/viz/theme.py b/viz/theme.py
index 6c4d68f..bc588e0 100644
--- a/viz/theme.py
+++ b/viz/theme.py
@@ -32,4 +32,8 @@ PAGE_COLORS = {
     7: "#FD79A8",  # Conv Block — hồng
     8: "#E17055",  # Full Pipeline — cam đất
+    9: "#00CEC9",  # BatchNorm — ngọc lam
+    10: "#0984E3",  # Softmax + Cross-Entropy — xanh đậm
+    11: "#E84393",  # Backprop & Training — hồng đậm
+    12: "#8E44AD",  # Generalization — tím đậm
 }
 
```

```diff
diff --git a/viz/widgets.py b/viz/widgets.py
index bcf8f4b..37224e8 100644
--- a/viz/widgets.py
+++ b/viz/widgets.py
@@ -9,4 +9,6 @@ import pandas as pd
 import streamlit as st
 
+from cnn_core.layers import Layer
+from cnn_core.train import TrainConfig, train
 from viz.theme import PRIMARY, step_dots
 
@@ -62,2 +64,13 @@ def run_or_hint(fn: Callable, *args: Any, todo_hint: str, **kwargs: Any):
         st.error(f"Lỗi khi chạy `{fn.__name__}`: {exc}")
         st.stop()
+
+
+@st.cache_resource(show_spinner="Đang train mạng bằng numpy thuần…", max_entries=64)
+def cached_train(cfg: TrainConfig) -> tuple[list[Layer], dict[str, list]]:
+    """train(cfg) có cache theo config — đổi qua lại giữa các trang/tuỳ chọn
+    không phải train lại. Dùng chung cho trang 9–12.
+
+    Model trả về là object DÙNG CHUNG giữa các lần gọi: chỉ gọi forward với
+    train=False trên nó (không update running stats / không train tiếp).
+    """
+    return train(cfg)
```

- [ ] **Step 6: Verify**

Run: `pytest -q`
Expected: every existing test plus the new block test passes (`43 passed`).

Run this helper smoke check. It also checks the symmetric colour scale from Review Focus #4:

```bash
python -c "
import matplotlib; matplotlib.use('Agg'); import numpy as np
from viz.matrix_view import *
f = fig_feature_maps(np.array([[[-2.0, 1.0]]]), shared_scale=True)
assert f.axes[0].images[0].get_clim() == (-2.0, 2.0)
fig_activation_hist([('a', np.random.randn(99))]); fig_lines({'s': [1, 2]}, xlabels=['a', 'b'], logy=True)
fig_curves([('r', {'train_acc': [.5, .9], 'val_acc': [.4, .8]})]); fig_bars(np.array([1., -2.]), highlight=1)
print('ok')"
```
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add viz/matrix_view.py cnn_core/block.py viz/theme.py viz/widgets.py tests/test_block.py
git commit -m "Add plotting helpers, ConvBlock.forward_steps and cached training

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Page smoke test + honest pages 7 and 8

**Files:**
- Create: `tests/test_pages.py`
- Modify: `pages/7_Conv_Block.py`
- Modify: `pages/8_Full_Pipeline.py`

**Interfaces:**
- Consumes: `ConvBlock.forward_steps`, `fig_feature_maps(shared_scale=...)`, `fig_activation_hist`, `fig_lines`.
- Produces: `tests/test_pages.py`. It globs `pages/*.py`, so every later page is covered automatically.

- [ ] **Step 1: Write the smoke test** — create `tests/test_pages.py`:

```python
"""Smoke test: mọi trang Streamlit render hết không có exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("path", PAGES, ids=[p.name for p in PAGES])
def test_page_renders_without_exception(path):
    at = AppTest.from_file(str(path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_pages.py -q`
Expected: `9 passed`. The existing pages already render, so this is the regression baseline.

- [ ] **Step 3: Page 7** — shared scale toggle and per-step histogram. Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/pages/7_Conv_Block.py b/pages/7_Conv_Block.py
index 5ea30e3..2e52ae5 100644
--- a/pages/7_Conv_Block.py
+++ b/pages/7_Conv_Block.py
@@ -4,5 +4,5 @@ import streamlit as st
 
 from cnn_core.block import ConvBlock, make_random_kernels
-from viz.matrix_view import fig_feature_maps
+from viz.matrix_view import fig_activation_hist, fig_feature_maps
 from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
 from viz.widgets import run_or_hint
@@ -65,15 +65,30 @@ block = run_or_hint(
     todo_hint="Implement `ConvBlock.__init__` trong `cnn_core/block.py` (Phase 5).",
 )
-output = run_or_hint(
-    block.forward,
+steps = run_or_hint(
+    block.forward_steps,
     image,
     todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5) — cần conv2d_multichannel, relu, max_pool2d đã xong trước đó.",
 )
+output = steps[-1][1]
 
 with st.container(border=True):
     st.markdown(f"#### Feature maps sau Conv → ReLU → Pool: `{output.shape}`")
-    st.pyplot(fig_feature_maps(output, titles=[f"kernel {i}" for i in range(num_kernels)]))
+    shared = st.toggle(
+        "Chung thang màu cho mọi kernel",
+        value=True,
+        help="Tắt: mỗi map tự co giãn màu — dễ nhìn hình dạng nhưng che mất độ lớn thật.",
+    )
+    st.pyplot(fig_feature_maps(output, titles=[f"kernel {i}" for i in range(num_kernels)], shared_scale=shared))
     st.caption(
         f"Input {image.shape} -> {num_kernels} kernel {kernel_size}x{kernel_size} -> "
         f"output {output.shape}. Mỗi ô là 1 feature map — kernel khác nhau bắt pattern khác nhau."
     )
+
+with st.container(border=True):
+    st.markdown("#### Phân phối giá trị qua từng bước")
+    st.pyplot(fig_activation_hist([("input", image)] + steps, color=COLOR))
+    st.caption(
+        "Sau Conv: số âm lẫn dương quanh 0. Sau ReLU: toàn bộ phần âm dồn về đúng 0 "
+        "(cột cao ở 0, xem % zero). Sau Max Pool: chỉ giữ số lớn nhất mỗi vùng nên "
+        "phân phối dịch sang phải, ít số 0 hơn."
+    )
```

- [ ] **Step 4: Page 8** — conv/relu/pool sub-steps, shared-scale maps, std and %-zero depth charts, callout to page 9. Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/pages/8_Full_Pipeline.py b/pages/8_Full_Pipeline.py
index b290bd0..774f499 100644
--- a/pages/8_Full_Pipeline.py
+++ b/pages/8_Full_Pipeline.py
@@ -5,5 +5,5 @@ from PIL import Image
 
 from cnn_core.block import ConvBlock, make_random_kernels
-from viz.matrix_view import fig_feature_maps
+from viz.matrix_view import fig_feature_maps, fig_lines
 from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
 from viz.widgets import run_or_hint, step_controls
@@ -80,18 +80,26 @@ for i in range(num_blocks):
 
 # Forward toàn bộ trước (để có sẵn dữ liệu), nhưng CHỈ hiển thị tới đúng
-# block mà người dùng đã "đi" tới qua step_controls bên dưới.
+# bước mà người dùng đã "đi" tới qua step_controls bên dưới. Mỗi block tách
+# thành 3 bước con conv / relu / pool.
 activations = [("input", image)]
 x = image
 for i, block in enumerate(blocks):
-    x = run_or_hint(
-        block.forward,
+    steps = run_or_hint(
+        block.forward_steps,
         x,
         todo_hint="Implement `ConvBlock.forward` trong `cnn_core/block.py` (Phase 5).",
     )
-    activations.append((f"block {i + 1}", x))
+    activations += [(f"block {i + 1} · {name}", out) for name, out in steps]
+    x = steps[-1][1]
 
-st.markdown("#### Forward pass — đi qua từng block một")
+STEP_NOTES = {
+    "conv": "Mỗi kernel nhân-cộng trên MỌI kênh input rồi cộng lại → 1 map. Có cả số âm.",
+    "relu": "Cắt hết số âm về 0 — nhiều vùng tối hẳn.",
+    "pool": "Giữ số lớn nhất mỗi ô 2×2 → kích thước còn một nửa.",
+}
+
+st.markdown("#### Forward pass — đi qua từng bước của từng block")
 with st.container(border=True):
-    step = step_controls("pipeline_walk", len(activations), color=COLOR, step_label="Lớp")
+    step = step_controls("pipeline_walk", len(activations), color=COLOR, step_label="Bước")
     name, activation = activations[step]
 
@@ -103,6 +111,30 @@ with st.container(border=True):
         st.pyplot(fig)
     else:
-        st.markdown(f"**Block {step}** — shape `{activation.shape}` (sau Conv → ReLU → Pool)")
-        st.pyplot(fig_feature_maps(activation, titles=[f"k{c}" for c in range(activation.shape[-1])], max_cols=4))
+        kind = name.split(" · ")[1]
+        st.markdown(f"**{name}** — shape `{activation.shape}` · {STEP_NOTES[kind]}")
+        st.pyplot(
+            fig_feature_maps(
+                activation, titles=[f"k{c}" for c in range(activation.shape[-1])], max_cols=4, shared_scale=True
+            )
+        )
+        st.caption(
+            f"Thang màu chung cho cả bước này: min {activation.min():.3g} · max {activation.max():.3g} · "
+            f"std {activation.std():.3g}. So với bước trước để thấy độ lớn thay đổi thế nào."
+        )
+
+st.divider()
+st.markdown("#### Độ lớn activation qua độ sâu (toàn bộ pipeline)")
+names = [n for n, _ in activations]
+c1, c2 = st.columns(2)
+c1.pyplot(fig_lines({"std": [float(a.std()) for _, a in activations]}, xlabels=names, ylabel="std"))
+c2.pyplot(fig_lines({"% số 0": [100 * float(np.mean(a == 0)) for _, a in activations]}, xlabels=names, ylabel="% = 0"))
+callout(
+    "Kernel random N(0, 0.1) làm std co lại ~½ sau mỗi block, và ReLU tắt ngày càng "
+    "nhiều neuron. Xếp thêm vài chục lớp, tín hiệu gần như biến mất (hoặc nổ tung nếu "
+    "kernel lớn) — gradient lúc train cũng vậy. Đây chính là bài toán mà <b>Batch "
+    "Normalization</b> (trang 9) sinh ra để giải.",
+    color=COLOR,
+    label="Nhìn kỹ",
+)
 
 st.divider()
```

- [ ] **Step 5: Verify pages and the walk-through**

Run: `pytest tests/test_pages.py -q` → `9 passed`

Run:
```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/8_Full_Pipeline.py", default_timeout=120).run()
at.slider[0].set_value(3).run()
while not (btn := [b for b in at.button if b.key == "pipeline_walk_next"][0]).disabled:
    btn.click().run(); assert not at.exception, at.exception
print([m.value[:30] for m in at.markdown if m.value.startswith("**block")])
EOF
```
Expected: `['**block 3 · pool** — shape `(2, ']` with no assertion error.

- [ ] **Step 6: Commit**

```bash
git add tests/test_pages.py pages/7_Conv_Block.py pages/8_Full_Pipeline.py
git commit -m "Show real activation magnitudes and per-step pipeline on pages 7-8

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Page 9 — BatchNorm

**Files:**
- Create: `pages/9_BatchNorm.py`

**Interfaces:**
- Consumes: `BatchNorm2D`, `Conv2D`, `ReLU`, `TrainConfig`, `load_digits_split`, `cached_train`, `fig_activation_hist`, `fig_curves`, `fig_lines`, `fig_matrix`, `editable_matrix`, `formula_box`, `PAGE_COLORS[9]`.
- Produces: page only.

- [ ] **Step 1: Create the page** — `pages/9_BatchNorm.py`:

```python
import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.layers import BatchNorm2D, Conv2D, ReLU
from cnn_core.train import TrainConfig, load_digits_split
from viz.matrix_view import DIV, fig_activation_hist, fig_curves, fig_lines, fig_matrix
from viz.theme import PAGE_COLORS, callout, formula_box, hero, inject_base_css
from viz.widgets import cached_train, editable_matrix

COLOR = PAGE_COLORS[9]

st.set_page_config(page_title="BatchNorm — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="9. Batch Normalization",
    subtitle="Ép mỗi kênh về mean 0 / std 1 trên từng mini-batch — để tín hiệu không tắt dần hay nổ tung qua độ sâu.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Trang 8 cho thấy std activation co lại ~½ sau mỗi block. Khi train, trọng số "
    "mỗi lớp thay đổi liên tục nên phân phối input của lớp sau cũng trôi theo — lớp "
    "sau phải liên tục thích nghi lại. BatchNorm chuẩn hoá lại phân phối đó sau "
    "mỗi lớp, rồi cho mạng tự học scale (γ) và shift (β) phù hợp.",
    color=COLOR,
    label="Vì sao quan trọng",
)

x_train, _, _, _ = load_digits_split()

# ---------------------------------------------------------------------------
# (a) Từng bước tính trên 1 mini-batch nhỏ
# ---------------------------------------------------------------------------
st.markdown("#### (a) BN tính gì — trên 1 mini-batch 6 mẫu × 3 kênh")
with st.container(border=True):
    rng = np.random.default_rng(0)
    default_batch = np.round(
        np.column_stack([rng.normal(5, 2, 6), rng.normal(-3, 0.5, 6), rng.normal(0, 10, 6)]), 1
    )
    c_in, c_ctrl = st.columns([2, 1])
    with c_in:
        st.caption("Mỗi hàng = 1 mẫu, mỗi cột = 1 kênh. 3 kênh cố ý có thang đo rất khác nhau — sửa tuỳ ý.")
        batch = editable_matrix("bn_batch", default_batch)
    with c_ctrl:
        gamma = st.slider("γ (scale học được)", 0.1, 3.0, 1.0, 0.1)
        beta = st.slider("β (shift học được)", -3.0, 3.0, 0.0, 0.1)

    bn = BatchNorm2D(3)
    bn.params["gamma"][:] = gamma
    bn.params["beta"][:] = beta
    out = bn.forward(batch, train=True)
    mean, var = batch.mean(axis=0), batch.var(axis=0)
    x_hat = (batch - mean) / np.sqrt(var + bn.eps)

    st.table(
        pd.DataFrame(
            {"μ (mean kênh)": mean, "σ² (var kênh)": var, "mean sau BN": out.mean(axis=0), "std sau BN": out.std(axis=0)},
            index=[f"kênh {c}" for c in range(3)],
        ).round(3)
    )
    formula_box(
        f"x̂ = (x − μ) / √(σ² + ε) &nbsp;&nbsp;→&nbsp;&nbsp; ô [0, 0]: ({batch[0, 0]:.2f} − {mean[0]:.2f}) / "
        f"√({var[0]:.2f} + 1e-5) = {x_hat[0, 0]:.3f}<br>"
        f"y = γ·x̂ + β = {gamma:.1f}·{x_hat[0, 0]:.3f} + {beta:.1f} = {out[0, 0]:.3f}",
        color=COLOR,
    )
    c1, c2, c3 = st.columns(3)
    c1.pyplot(fig_matrix(batch, title="x (input)", cmap=DIV))
    c2.pyplot(fig_matrix(x_hat, title="x̂ (chuẩn hoá)", cmap=DIV))
    c3.pyplot(fig_matrix(out, title="y = γx̂ + β", cmap=DIV))
    st.caption(
        "Sau BN mỗi cột có mean = β và std = γ, bất kể thang đo ban đầu. Nếu γ = σ và β = μ "
        "thì BN trả lại đúng input — nghĩa là BN không làm mất khả năng biểu diễn, chỉ cho "
        "mạng một cách 'dễ điều khiển' hơn để chọn phân phối."
    )

# ---------------------------------------------------------------------------
# (b) Stack nhiều conv random: có vs không BN
# ---------------------------------------------------------------------------
st.markdown("#### (b) Vì sao cần — stack nhiều lớp conv random, có vs không BN")
with st.container(border=True):
    c1, c2 = st.columns(2)
    depth = c1.slider("Số lớp conv", 2, 10, 6)
    init_name = c2.selectbox(
        "Khởi tạo trọng số",
        ["Nhỏ (He × 0.5)", "He (chuẩn)", "Lớn (He × 2)"],
        help="He init = std √(2/fan_in), thiết kế để giữ std ổn định qua ReLU.",
    )
    scale = {"Nhỏ (He × 0.5)": 0.5, "He (chuẩn)": 1.0, "Lớn (He × 2)": 2.0}[init_name]

    rng = np.random.default_rng(1)
    convs = []
    in_ch = 1
    for _ in range(depth):
        conv = Conv2D(in_ch, 8, 3, padding=1, rng=rng)
        conv.params["W"] *= scale
        convs.append(conv)
        in_ch = 8

    batch_imgs = x_train[:64]
    results = {}
    for use_bn in (False, True):
        x, per_layer = batch_imgs, []
        for i, conv in enumerate(convs):
            x = conv.forward(x)
            if use_bn:
                x = BatchNorm2D(8).forward(x, train=True)
            x = ReLU().forward(x)
            per_layer.append((f"lớp {i + 1}", x))
        results["Có BN" if use_bn else "Không BN"] = per_layer

    names = [n for n, _ in results["Không BN"]]
    st.pyplot(
        fig_lines(
            {label: [float(a.std()) for _, a in layers] for label, layers in results.items()},
            xlabels=names,
            ylabel="std activation (log)",
            logy=True,
        )
    )
    tab_off, tab_on = st.tabs(["Histogram — không BN", "Histogram — có BN"])
    with tab_off:
        st.pyplot(fig_activation_hist(results["Không BN"], color="#E17055", max_cols=5))
    with tab_on:
        st.pyplot(fig_activation_hist(results["Có BN"], color=COLOR, max_cols=5))
    st.caption(
        "Không BN: std nhân lên đều đặn qua mỗi lớp (×0.5 → co về ~0, ×2 → nổ), nên sau "
        "vài chục lớp tín hiệu và gradient đều hỏng. He init đúng giữ được std lúc khởi "
        "tạo, nhưng trong lúc train trọng số trôi đi. Có BN: std mỗi lớp luôn được kéo về "
        "cùng mức, bất kể khởi tạo."
    )

# ---------------------------------------------------------------------------
# (c) Train vs eval
# ---------------------------------------------------------------------------
st.markdown("#### (c) Train mode vs eval mode — batch stats vs running stats")
with st.container(border=True):
    batch_size = st.select_slider("Batch size", options=[2, 4, 8, 16, 32, 64, 128], value=8)
    conv = Conv2D(1, 4, 3, padding=1, rng=np.random.default_rng(2))
    feats = conv.forward(x_train)  # (N, 8, 8, 4)
    population_mean = float(feats[..., 0].mean())

    bn = BatchNorm2D(4)
    rng = np.random.default_rng(3)
    batch_means, running = [], []
    for _ in range(60):
        idx = rng.choice(len(feats), batch_size, replace=False)
        batch_means.append(float(feats[idx][..., 0].mean()))
        bn.forward(feats[idx], train=True)
        running.append(float(bn.running_mean[0]))

    st.pyplot(
        fig_lines(
            {"μ của từng batch": batch_means, "running mean": running, "mean toàn bộ data": [population_mean] * 60},
            xlabel="batch thứ",
            ylabel="mean kênh 0",
        )
    )
    st.caption(
        f"Batch nhỏ → μ mỗi batch dao động mạnh quanh mean thật (độ lệch ∝ 1/√batch). "
        f"Running mean = trung bình trượt (momentum {bn.momentum}) của các μ đó, hội tụ về "
        f"mean thật. Nhiễu batch này cũng là lý do BN có chút tác dụng regularize."
    )

    image = x_train[:1]
    others_a, others_b = x_train[10:17], x_train[100:107]
    bn_trained = BatchNorm2D(4)
    for i in range(0, len(feats), 64):
        bn_trained.forward(feats[i : i + 64], train=True)
    f_img, f_a, f_b = conv.forward(image), conv.forward(others_a), conv.forward(others_b)
    out_a = bn_trained.forward(np.concatenate([f_img, f_a]), train=True)[0]
    out_b = bn_trained.forward(np.concatenate([f_img, f_b]), train=True)[0]
    eval_a = bn_trained.forward(np.concatenate([f_img, f_a]), train=False)[0]
    eval_b = bn_trained.forward(np.concatenate([f_img, f_b]), train=False)[0]
    c1, c2 = st.columns(2)
    c1.metric("Train mode: cùng 1 ảnh, 2 batch khác nhau — chênh lệch output", f"{np.abs(out_a - out_b).max():.3f}")
    c2.metric("Eval mode: cùng 1 ảnh, 2 batch khác nhau — chênh lệch output", f"{np.abs(eval_a - eval_b).max():.3f}")
    callout(
        "Ở train mode, output của 1 ảnh phụ thuộc vào các ảnh <i>khác</i> cùng batch. Lúc "
        "dự đoán, 1 ảnh phải luôn cho cùng 1 kết quả (và có khi chỉ có 1 ảnh) — nên eval "
        "dùng running stats cố định đã tích luỹ lúc train. Quên chuyển sang eval mode là "
        "bug kinh điển (<code>model.eval()</code> trong PyTorch).",
        color=COLOR,
        label="Hệ quả",
    )

# ---------------------------------------------------------------------------
# (d) Hiệu quả khi train thật
# ---------------------------------------------------------------------------
st.markdown("#### (d) Khi train thật — BN giúp tối ưu nhanh hơn")
with st.container(border=True):
    epochs = st.slider("Số epoch", 3, 20, 10, key="bn_epochs")
    runs = []
    for use_bn in (False, True):
        _, history = cached_train(TrainConfig(use_bn=use_bn, epochs=epochs))
        runs.append(("Có BN" if use_bn else "Không BN", history))
    c1, c2 = st.columns(2)
    c1.pyplot(fig_curves(runs, "acc"))
    c2.pyplot(fig_curves(runs, "loss"))
    st.caption(
        f"Cùng kiến trúc, cùng khởi tạo, cùng lr. Sau epoch 1: val acc "
        f"{runs[0][1]['val_acc'][0]:.2f} (không BN) vs {runs[1][1]['val_acc'][0]:.2f} (có BN). "
        f"BN chủ yếu giúp TỐI ƯU dễ hơn — tác dụng chống overfit của nó nhỏ (xem trang 12)."
    )
```

- [ ] **Step 2: Verify render + train/eval metric**

Run: `pytest tests/test_pages.py -q -k BatchNorm` → `1 passed`

Run:
```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/9_BatchNorm.py", default_timeout=120).run()
at.selectbox[0].set_value("Lớn (He × 2)").run()
assert not at.exception
print([m.value for m in at.metric])
EOF
```
Expected: the first metric is > 0 (train mode depends on the batch), the second is `0.000`.

- [ ] **Step 3: Commit**

```bash
git add pages/9_BatchNorm.py
git commit -m "Add BatchNorm page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Page 10 — Softmax + Cross-Entropy

**Files:**
- Create: `pages/10_Softmax_Cross_Entropy.py`

**Interfaces:**
- Consumes: `softmax`, `cross_entropy`, `softmax_cross_entropy_backward`, `TrainConfig`, `forward_trace`, `load_digits_split`, `cached_train`, `step_controls`, `fig_bars`, `fig_feature_maps`, `PAGE_COLORS[10]`.
- Produces: page only.

- [ ] **Step 1: Create the page** — `pages/10_Softmax_Cross_Entropy.py`:

```python
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.layers import cross_entropy, softmax, softmax_cross_entropy_backward
from cnn_core.train import TrainConfig, forward_trace, load_digits_split
from viz.matrix_view import DIV, fig_bars, fig_feature_maps
from viz.theme import ACCENT, BORDER, PAGE_COLORS, TEXT, callout, formula_box, hero, inject_base_css
from viz.widgets import cached_train, step_controls

COLOR = PAGE_COLORS[10]

st.set_page_config(page_title="Softmax + Cross-Entropy — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="10. Softmax + Cross-Entropy",
    subtitle="Biến 10 con số thô (logits) thành xác suất, rồi đo mạng 'sai' đến mức nào bằng 1 con số duy nhất.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Lớp cuối của CNN chỉ nhả ra 10 số thực bất kỳ (logits). Softmax biến chúng "
    "thành xác suất (dương, tổng = 1). Cross-entropy = −log(xác suất gán cho lớp "
    "đúng): đoán đúng chắc chắn → loss ≈ 0, đoán sai chắc chắn → loss rất lớn. "
    "Đây là con số mà toàn bộ quá trình train cố kéo xuống.",
    color=COLOR,
    label="Vì sao quan trọng",
)

# ---------------------------------------------------------------------------
# (a) Tự kéo logits
# ---------------------------------------------------------------------------
st.markdown("#### (a) Tự kéo logits — xem xác suất, loss và gradient đổi theo")
with st.container(border=True):
    n_classes = 5
    defaults = [2.0, 1.0, 0.5, -1.0, 0.0]
    cols = st.columns(n_classes + 1)
    logits = np.array([cols[i].slider(f"z{i}", -5.0, 5.0, defaults[i], 0.1) for i in range(n_classes)])
    true_class = cols[-1].radio("Lớp đúng", list(range(n_classes)), index=0)

    probs = softmax(logits)
    loss = cross_entropy(probs[None], np.array([true_class]))
    grad = softmax_cross_entropy_backward(probs[None], np.array([true_class]))[0]
    labels = [str(i) for i in range(n_classes)]

    c1, c2, c3 = st.columns(3)
    c1.pyplot(fig_bars(logits, labels, highlight=true_class, title="Logits z (thô)", color=COLOR))
    c2.pyplot(fig_bars(probs, labels, highlight=true_class, title="Softmax p (tổng = 1)", color=COLOR))
    c3.pyplot(fig_bars(grad, labels, highlight=true_class, title="Gradient ∂L/∂z = p − y", color=COLOR))

    exps = " + ".join(f"e^{z:.1f}" for z in logits)
    formula_box(
        f"p{true_class} = e^{logits[true_class]:.1f} / ({exps}) = {probs[true_class]:.3f}<br>"
        f"L = −log p{true_class} = −log({probs[true_class]:.3f}) = {loss:.3f}",
        color=COLOR,
    )

    c_curve, c_text = st.columns([1.2, 1])
    with c_curve:
        p = np.linspace(0.005, 1, 200)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(p, -np.log(p), color=COLOR, linewidth=2.5)
        ax.scatter([probs[true_class]], [loss], color=ACCENT, s=80, zorder=3, edgecolor=TEXT)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax.set_xlabel("p (xác suất gán cho lớp đúng)")
        ax.set_ylabel("loss = −log p")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
    with c_text:
        st.markdown(
            "- Đường cong **dốc đứng khi p → 0**: đoán sai mà còn tự tin bị phạt cực nặng "
            "(p = 0.01 → loss 4.6), trong khi từ 0.9 lên 0.99 chỉ bớt được 0.09.\n"
            "- Gradient **p − y** gọn đến bất ngờ: logit lớp đúng bị đẩy *lên* một lượng "
            "(1 − p), mọi logit khác bị đẩy *xuống* đúng bằng xác suất của nó.\n"
            "- Cộng cùng 1 số vào mọi logit → softmax không đổi. Chỉ **chênh lệch** giữa "
            "các logit là quan trọng."
        )

# ---------------------------------------------------------------------------
# (b) 1 ảnh thật đi hết mạng tới loss
# ---------------------------------------------------------------------------
st.markdown("#### (b) 1 ảnh chữ số đi hết mạng đã train, tới tận loss")
with st.container(border=True):
    model, _ = cached_train(TrainConfig())
    _, _, x_val, y_val = load_digits_split()
    idx = st.slider("Chọn ảnh trong tập val", 0, len(x_val) - 1, 0)
    image, label = x_val[idx : idx + 1], int(y_val[idx])

    trace = forward_trace(model, image)
    probs = softmax(trace[-1][1])
    loss = cross_entropy(probs, np.array([label]))
    stages = trace + [("Softmax", probs), ("Cross-Entropy", np.array([[loss]]))]

    step = step_controls("ce_trace", len(stages), color=COLOR, step_label="Lớp")
    name, out = stages[step]
    st.markdown(f"**{name}** — shape `{out.shape[1:]}` · nhãn đúng: **{label}**")

    digits = [str(d) for d in range(10)]
    if name == "Input":
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(out[0, :, :, 0], cmap="gray_r")
        ax.axis("off")
        st.pyplot(fig)
    elif out.ndim == 4:
        st.pyplot(fig_feature_maps(out[0], max_cols=8, shared_scale=True, cmap=DIV if name == "Conv2D" else "magma"))
    elif name == "Cross-Entropy":
        formula_box(f"L = −log p{label} = −log({probs[0, label]:.4f}) = {loss:.4f}", color=COLOR)
    elif name == "Softmax":
        st.pyplot(fig_bars(out[0], digits, highlight=label, title="xác suất", color=COLOR, figsize=(6, 2.8)))
    elif step == len(trace) - 1:
        st.pyplot(fig_bars(out[0], digits, highlight=label, title="logits", color=COLOR, figsize=(6, 2.8)))
    else:
        st.pyplot(fig_bars(out[0], title=f"vector {out.shape[1]} chiều", color=COLOR, figsize=(6, 2.4)))

    pred = int(probs.argmax())
    st.caption(
        f"Mạng đoán **{pred}** với xác suất {probs[0, pred]:.1%} — "
        + ("đúng." if pred == label else f"SAI (đúng là {label}). Thử bấm tới bước Softmax xem mạng phân vân giữa các lớp nào.")
    )
```

- [ ] **Step 2: Verify every trace stage renders**

Run:
```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/10_Softmax_Cross_Entropy.py", default_timeout=120).run()
while not (btn := [b for b in at.button if b.key == "ce_trace_next"][0]).disabled:
    btn.click().run(); assert not at.exception, at.exception
print([m.value for m in at.markdown if m.value.startswith("**")][-1])
EOF
```
Expected: the last line starts with `**Cross-Entropy**`, with no assertion error.

- [ ] **Step 3: Commit**

```bash
git add pages/10_Softmax_Cross_Entropy.py
git commit -m "Add softmax and cross-entropy page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Page 11 — Backprop & Training

**Files:**
- Create: `pages/11_Backprop_Training.py`

**Interfaces:**
- Consumes: `TrainConfig`, `build_model`, `forward_trace`, `load_digits_split`, `param_layer_names`, `cached_train`, `fig_curves`, `fig_feature_maps`, `fig_lines`, `PAGE_COLORS[11]`.
- Produces: page only.

- [ ] **Step 1: Create the page** — `pages/11_Backprop_Training.py`:

```python
import numpy as np
import streamlit as st

from cnn_core.train import TrainConfig, build_model, forward_trace, load_digits_split, param_layer_names
from viz.matrix_view import DIV, fig_curves, fig_feature_maps, fig_lines
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[11]

st.set_page_config(page_title="Backprop & Training — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="11. Backprop & Training",
    subtitle="Kernel không còn random: lan truyền ngược gradient của loss về từng lớp và sửa trọng số từng chút một.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Trang 1–8 dùng kernel random. Train = lặp lại: forward → loss → backward (chain "
    "rule: mỗi lớp nhận ∂L/∂output, trả về ∂L/∂input cho lớp trước và ∂L/∂W cho chính "
    "nó) → trừ gradient × lr vào trọng số. Sau vài nghìn bước như vậy, kernel tự biến "
    "thành các bộ dò nét/cạnh có ý nghĩa.",
    color=COLOR,
    label="Vì sao quan trọng",
)

with st.expander("Toàn bộ vòng lặp train trong `cnn_core/train.py` (rút gọn)", expanded=False):
    st.code(
        """for xb, yb in batches:
    probs = softmax(forward(model, xb, train=True))       # forward
    dout = softmax_cross_entropy_backward(probs, yb)       # ∂L/∂logits = (p − y)/N
    for layer in reversed(model):                          # backward: chain rule
        dout = layer.backward(dout)                        #   lớp tự lưu grads["W"]
    for layer in model:                                    # SGD + momentum
        for k, param in layer.params.items():
            vel[k] = 0.9 * vel[k] - lr * layer.grads[k]
            param += vel[k]""",
        language="python",
    )

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    epochs = c1.slider("Số epoch", 1, 30, 15)
    lr = c2.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.01)
    use_bn = c3.toggle("BatchNorm", value=True)

cfg = TrainConfig(use_bn=use_bn, epochs=epochs, lr=lr)
model, history = cached_train(cfg)

if not np.isfinite(history["train_loss"][-1]):
    st.error(
        f"Loss thành NaN/∞ — lr = {lr} quá lớn: mỗi bước nhảy vượt quá đáy rồi văng ra xa hơn, "
        "trọng số nổ tung. Giảm lr xuống."
    )

st.markdown("#### Đường học")
c1, c2 = st.columns(2)
c1.pyplot(fig_curves([(f"lr={lr}", history)], "loss"))
c2.pyplot(fig_curves([(f"lr={lr}", history)], "acc"))
st.caption(
    f"Val acc cuối: {history['val_acc'][-1]:.3f}. Thử lr 0.001 (học quá chậm) và 0.3 "
    "(dao động / nổ) để thấy vì sao chọn learning rate là tham số quan trọng nhất."
)

st.markdown("#### Gradient chảy về từng lớp")
names = param_layer_names(model)
st.pyplot(
    fig_lines(
        {name: [g[i] for g in history["grad_norms"]] for i, name in enumerate(names)},
        xlabel="epoch",
        ylabel="‖∂L/∂W‖ trung bình (log)",
        logy=True,
    )
)
st.caption(
    "Gradient lớn lúc đầu (mạng sai nhiều) rồi nhỏ dần khi loss giảm. Tắt BatchNorm "
    "để so độ lớn gradient giữa các lớp — lớp nào nhận gradient quá nhỏ thì học rất chậm."
)

st.markdown("#### Trước vs sau khi train")
init_model = build_model(cfg)  # cùng seed → đúng trọng số lúc khởi tạo của lần train này
with st.container(border=True):
    st.markdown("**16 kernel 3×3 của lớp Conv đầu tiên**")
    c1, c2 = st.columns(2)
    c1.caption("Lúc khởi tạo (random)")
    c1.pyplot(fig_feature_maps(init_model[0].params["W"][:, :, 0, :], max_cols=8, cmap=DIV, shared_scale=True))
    c2.caption("Sau khi train")
    c2.pyplot(fig_feature_maps(model[0].params["W"][:, :, 0, :], max_cols=8, cmap=DIV, shared_scale=True))

with st.container(border=True):
    _, _, x_val, y_val = load_digits_split()
    idx = st.slider("Ảnh val", 0, len(x_val) - 1, 3)
    image = x_val[idx : idx + 1]
    st.markdown(f"**Feature map sau Conv 1 → ReLU của ảnh chữ số {y_val[idx]}**")

    def first_relu(m):
        return next(out for name, out in forward_trace(m, image) if name == "ReLU")[0]

    c1, c2 = st.columns(2)
    c1.caption("Lúc khởi tạo")
    c1.pyplot(fig_feature_maps(first_relu(init_model), max_cols=8, cmap="magma", shared_scale=True))
    c2.caption("Sau khi train")
    c2.pyplot(fig_feature_maps(first_relu(model), max_cols=8, cmap="magma", shared_scale=True))
    st.caption(
        "Sau khi train, nhiều kernel trở thành bộ dò nét ngang / dọc / chéo cụ thể — mỗi map "
        "sáng lên ở đúng phần nét của chữ số mà kernel đó 'thích'."
    )
```

- [ ] **Step 2: Verify, including the diverging-lr path (Review Focus #1)**

Run:
```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/11_Backprop_Training.py", default_timeout=120).run()
at.toggle[0].set_value(False).run()
at.select_slider[0].set_value(0.3).run()
assert not at.exception, at.exception
print("errors:", [e.value[:30] for e in at.error])
EOF
```
Expected: no exception. If this run diverges, `errors:` lists the "Loss thành NaN/∞" message. If it does not diverge, the list is empty, which is also fine. The page must not crash either way.

- [ ] **Step 3: Commit**

```bash
git add pages/11_Backprop_Training.py
git commit -m "Add backprop and training page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Page 12 — Generalization

**Files:**
- Create: `pages/12_Generalization.py`

**Interfaces:**
- Consumes: `MAX_TRAIN_SIZE`, `TrainConfig`, `load_digits_split`, `predict`, `cached_train`, `fig_curves`, `fig_feature_maps`, `PAGE_COLORS[12]`.
- Produces: page only. Session keys: `g_preset`, `g_train_size`, `g_label_noise`, `g_dropout`, `g_weight_decay`, `g_augment`, `g_use_bn`, `g_epochs`, `g_runs` (list of `TrainConfig`).

- [ ] **Step 1: Create the page** — `pages/12_Generalization.py`:

```python
import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.train import MAX_TRAIN_SIZE, TrainConfig, load_digits_split, predict
from viz.matrix_view import fig_curves, fig_feature_maps
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[12]

st.set_page_config(page_title="Generalization — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="12. Generalization",
    subtitle="Train acc 100% chưa nói lên gì. Cái cần là đúng trên ảnh chưa từng thấy — và vì sao các kỹ thuật kéo đường val lên.",
    badge="Phần 2 · Train thật",
    color=COLOR,
)
callout(
    "Mạng đủ lớn có thể <b>học thuộc</b> tập train — kể cả nhãn sai. Lúc đó train loss "
    "tiếp tục giảm nhưng val loss quay đầu tăng: mạng đang học chi tiết riêng của từng "
    "ảnh train thay vì quy luật chung. Khoảng cách train − val (gap) là thước đo overfit.",
    color=COLOR,
    label="Vì sao quan trọng",
)

OVERFIT = dict(
    train_size=300, label_noise=0.3, dropout=0.0, weight_decay=0.0, augment=False, use_bn=False, epochs=60
)
PRESETS = {
    "1. Overfit: 300 ảnh, 30% nhãn sai": OVERFIT,
    "2. + Dropout 0.5": {**OVERFIT, "dropout": 0.5},
    "3. + Weight decay 0.02": {**OVERFIT, "weight_decay": 0.02},
    "4. + Augmentation (dịch ±1px)": {**OVERFIT, "augment": True},
    "5. + BatchNorm (không phải regularizer!)": {**OVERFIT, "use_bn": True},
    "6. + Nhiều data hơn (1297 ảnh)": {**OVERFIT, "train_size": MAX_TRAIN_SIZE},
    "7. Data sạch (0% nhãn sai)": {**OVERFIT, "label_noise": 0.0},
}


def apply_preset() -> None:
    for key, value in PRESETS[st.session_state["g_preset"]].items():
        st.session_state[f"g_{key}"] = value


def run_label(cfg: TrainConfig) -> str:
    parts = [f"{cfg.train_size} ảnh", f"noise {cfg.label_noise:.0%}"]
    if cfg.dropout:
        parts.append(f"drop {cfg.dropout}")
    if cfg.weight_decay:
        parts.append(f"wd {cfg.weight_decay}")
    if cfg.augment:
        parts.append("aug")
    if cfg.use_bn:
        parts.append("BN")
    return " · ".join(parts)


if "g_train_size" not in st.session_state:
    st.session_state["g_preset"] = next(iter(PRESETS))
    apply_preset()
if "g_runs" not in st.session_state:
    st.session_state["g_runs"] = [TrainConfig(**OVERFIT)]

with st.container(border=True):
    st.selectbox("Preset (chọn lần lượt 1 → 7 rồi bấm Chạy)", list(PRESETS), key="g_preset", on_change=apply_preset)
    c1, c2, c3 = st.columns(3)
    c1.select_slider("Số ảnh train", options=[50, 100, 200, 300, 500, 800, MAX_TRAIN_SIZE], key="g_train_size")
    c1.slider("Tỉ lệ nhãn sai", 0.0, 0.5, key="g_label_noise", step=0.1)
    c2.slider("Dropout", 0.0, 0.7, key="g_dropout", step=0.1)
    c2.select_slider("Weight decay", options=[0.0, 0.001, 0.005, 0.02, 0.05], key="g_weight_decay")
    c3.toggle("Augmentation (dịch ±1px)", key="g_augment")
    c3.toggle("BatchNorm", key="g_use_bn")
    c3.slider("Số epoch", 10, 100, key="g_epochs", step=10)

    cfg = TrainConfig(**{k: st.session_state[f"g_{k}"] for k in OVERFIT})
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Chạy & thêm vào so sánh", type="primary", use_container_width=True):
        if cfg not in st.session_state["g_runs"]:
            st.session_state["g_runs"].append(cfg)
    if b2.button("Xoá hết", use_container_width=True):
        st.session_state["g_runs"] = []

runs = st.session_state["g_runs"]
if not runs:
    st.info("Chưa có lần chạy nào — chọn preset rồi bấm **Chạy**.")
    st.stop()

results = [(run_label(c), *cached_train(c)) for c in runs]

st.markdown("#### Đường học — mọi lần chạy chồng lên nhau")
c1, c2 = st.columns(2)
c1.pyplot(fig_curves([(label, h) for label, _, h in results], "acc"))
c2.pyplot(fig_curves([(label, h) for label, _, h in results], "loss"))

rows = []
for label, _, h in results:
    best = int(np.argmax(h["val_acc"]))
    rows.append(
        {
            "Lần chạy": label,
            "Train acc": h["train_acc"][-1],
            "Val acc cuối": h["val_acc"][-1],
            "Val tốt nhất": h["val_acc"][best],
            "@ epoch": best + 1,
            "Gap (train − val)": h["train_acc"][-1] - h["val_acc"][-1],
        }
    )
st.dataframe(
    pd.DataFrame(rows).style.format(
        {c: "{:.3f}" for c in ["Train acc", "Val acc cuối", "Val tốt nhất", "Gap (train − val)"]}
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Val tốt nhất ≠ val cuối nghĩa là mạng đã qua đỉnh rồi bắt đầu overfit — dừng ở epoch "
    "đó (early stopping) là kỹ thuật regularize rẻ nhất. Train acc đo trên nhãn train (có "
    "nhiễu), val đo trên nhãn sạch — nên gap ÂM là dấu hiệu tốt: mạng từ chối học thuộc nhãn sai."
)

st.markdown("#### Vì sao từng kỹ thuật kéo đường val lên")
c1, c2 = st.columns(2)
with c1:
    callout(
        "Mỗi bước train tắt ngẫu nhiên p% neuron → mạng không thể dựa vào 1 neuron cụ "
        "thể để 'nhớ' 1 ảnh, buộc thông tin phải phân tán. Giống train nhiều mạng con "
        "rồi lấy trung bình (ensemble). Train acc thấp hơn, val cao hơn.",
        color=COLOR,
        label="Dropout",
    )
    callout(
        "Cộng λ·W vào gradient → kéo mọi trọng số về 0. Học thuộc nhiễu cần trọng số lớn "
        "để uốn hàm quanh từng điểm lẻ; phạt trọng số lớn → hàm mượt hơn, bỏ qua điểm lẻ.",
        color=COLOR,
        label="Weight decay (L2)",
    )
    callout(
        "Dịch ảnh ±1px mỗi lần → mạng không bao giờ thấy lại đúng 1 ảnh, nên không thể "
        "nhớ theo pixel. Là cách 'thêm data' miễn phí, mã hoá luôn hiểu biết của ta rằng "
        "chữ số dịch 1px vẫn là chữ số đó.",
        color=COLOR,
        label="Augmentation",
    )
with c2:
    callout(
        "BN giúp tối ưu nhanh hơn (trang 9) — nhưng tối ưu nhanh hơn cũng là học thuộc "
        "nhiễu nhanh hơn. Nhiễu từ batch stats chỉ regularize rất nhẹ. Đừng kỳ vọng BN "
        "chữa overfit.",
        color=COLOR,
        label="BatchNorm",
    )
    callout(
        "Nhiều ảnh thật hơn là cách chắc chắn nhất: quy luật chung lặp lại qua nhiều ảnh, "
        "còn nhiễu thì không — học thuộc trở nên đắt hơn học quy luật. Mọi kỹ thuật khác "
        "chỉ là cách giả lập điều này khi không có thêm data.",
        color=COLOR,
        label="Thêm data",
    )
    callout(
        "Nhãn sai là thứ duy nhất mạng CHỈ có thể học bằng cách học thuộc. Train acc vẫn "
        "leo lên gần 100% nghĩa là mạng đã nhớ cả nhãn sai — dấu hiệu overfit rõ nhất.",
        color=COLOR,
        label="Nhãn sai",
    )

st.markdown("#### Ảnh val bị đoán sai — lần chạy mới nhất")
label, model, _ = results[-1]
_, _, x_val, y_val = load_digits_split()
pred = predict(model, x_val).argmax(axis=1)
wrong = np.flatnonzero(pred != y_val)[:16]
st.caption(f"{label}: sai {int(np.sum(pred != y_val))}/{len(y_val)} ảnh val. 16 ảnh sai đầu tiên:")
if len(wrong):
    st.pyplot(
        fig_feature_maps(
            x_val[wrong, :, :, 0].transpose(1, 2, 0),
            titles=[f"đoán {pred[i]} · đúng {y_val[i]}" for i in wrong],
            max_cols=8,
            cmap="gray_r",
        )
    )
```

- [ ] **Step 2: Verify preset → run → clear (Review Focus #2)**

Run:
```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/12_Generalization.py", default_timeout=120).run()
at.selectbox(key="g_preset").set_value("2. + Dropout 0.5").run()
assert at.session_state["g_dropout"] == 0.5
at.button[0].click().run(); at.button[0].click().run()   # 2nd click must not duplicate
assert len(at.session_state["g_runs"]) == 2 and not at.exception
print(at.dataframe[0].value[["Lần chạy", "Val acc cuối", "Val tốt nhất"]])
at.button[1].click().run()
assert at.info and not at.exception
print("ok")
EOF
```
Expected: a 2-row table where the dropout run's "Val acc cuối" is higher than the baseline's (about 0.88 vs 0.78), then `ok`.

- [ ] **Step 3: Commit**

```bash
git add pages/12_Generalization.py
git commit -m "Add generalization page with run comparison

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Home page + docs

**Files:**
- Modify: `app.py` (cards 9–12, readiness check, intro copy)
- Modify: `README.md`, `TODO.md`

**Interfaces:**
- Consumes: `TrainConfig`, `build_model`, `forward`.

- [ ] **Step 1: Home page** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/app.py b/app.py
index 92cc695..599ec21 100644
--- a/app.py
+++ b/app.py
@@ -15,5 +15,5 @@ inject_base_css()
 hero(
     title="CNN Visualizer",
-    subtitle="Tự tay viết từng phép toán của CNN — kernel, padding, stride, pooling — rồi xem nó chạy thật, từng bước một.",
+    subtitle="Tự tay viết từng phép toán của CNN — kernel, padding, stride, pooling — rồi train thật, xem BatchNorm, loss và overfit hoạt động ra sao.",
     badge="Học CNN bằng cách tự xây",
 )
@@ -22,7 +22,9 @@ st.markdown(
     f"""
     <div style="color:{MUTED}; font-size:1.0rem; margin-bottom:1.4rem;">
-    Phần toán (<code>cnn_core/</code>) do <b>bạn</b> viết — app chỉ vẽ lại kết quả.
-    Đi đúng thứ tự 8 bước bên dưới, mỗi bước implement xong 1-2 hàm là mở khoá
-    ngay 1 trang trực quan. Chi tiết từng phase ở <code>TODO.md</code>.
+    <b>Phần 1 (trang 1–8)</b>: phần toán forward trong <code>cnn_core/</code> do <b>bạn</b> viết bằng
+    vòng lặp tay — mỗi bước implement xong 1-2 hàm là mở khoá 1 trang. Chi tiết ở <code>TODO.md</code>.<br>
+    <b>Phần 2 (trang 9–12)</b>: <code>cnn_core/layers.py</code> + <code>train.py</code> (numpy vectorized,
+    viết sẵn, có backward) train 1 CNN thật trên ảnh chữ số 8×8 — để thấy BatchNorm, loss,
+    backprop và overfit bằng số thật.
     </div>
     """,
@@ -89,4 +91,13 @@ except ImportError:
 ready[8] = ready[7]  # Full pipeline dùng chung ConvBlock với trang 7
 
+try:
+    from cnn_core.train import TrainConfig, build_model, forward
+
+    ready[9] = _check(lambda: forward(build_model(TrainConfig()), np.zeros((1, 8, 8, 1))))
+except ImportError:
+    ready[9] = False
+for num in (10, 11, 12):  # trang 9–12 dùng chung layers.py + train.py
+    ready[num] = ready[9]
+
 PAGES = [
     (1, "pages/1_Kernel_va_Convolution.py", "Kernel & Convolution", "Trượt kernel qua input, nhân-cộng từng vị trí."),
@@ -98,7 +109,11 @@ PAGES = [
     (7, "pages/7_Conv_Block.py", "Conv Block", "Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh."),
     (8, "pages/8_Full_Pipeline.py", "Full Pipeline", "Xếp nhiều block, forward 1 ảnh thật qua toàn bộ chuỗi."),
+    (9, "pages/9_BatchNorm.py", "BatchNorm", "Chuẩn hoá từng kênh để tín hiệu không tắt/nổ qua độ sâu."),
+    (10, "pages/10_Softmax_Cross_Entropy.py", "Softmax + CE", "Logits → xác suất → loss: mạng sai đến mức nào."),
+    (11, "pages/11_Backprop_Training.py", "Backprop & Training", "Gradient chảy ngược, kernel random thành bộ dò nét."),
+    (12, "pages/12_Generalization.py", "Generalization", "Vì sao dropout, weight decay, augmentation kéo val lên."),
 ]
 
-st.markdown("### Hành trình 8 bước")
+st.markdown("### Hành trình 12 bước")
 
 cols = st.columns(4)
```

- [ ] **Step 2: Docs** — Apply by saving the block to `/tmp/p.diff` and running `git apply /tmp/p.diff`, or edit by hand.

```diff
diff --git a/README.md b/README.md
index e1cfb9b..2ee80ab 100644
--- a/README.md
+++ b/README.md
@@ -28,5 +28,5 @@ thuộc.
 ## Tính năng
 
-App gồm 8 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học:
+App gồm 12 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học. Trang 1–8 (phần 1) là forward pass với kernel random; trang 9–12 (phần 2) train thật 1 CNN nhỏ để thấy BatchNorm, loss, backprop và overfit:
 
 | # | Trang | Nội dung |
@@ -38,6 +38,10 @@ App gồm 8 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo th
 | 5 | Pooling | So sánh trực quan max-pooling và average-pooling trên cùng 1 input |
 | 6 | Activation | ReLU cắt giá trị âm trên feature map; so sánh đường cong ReLU với Sigmoid |
-| 7 | Conv Block | Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh (RGB), quan sát nhiều feature map ra từ nhiều kernel |
-| 8 | Full Pipeline | Xếp 2-3 Conv Block liên tiếp, forward 1 ảnh thật (upload hoặc ảnh tổng hợp) qua từng block một, theo dõi shape thu nhỏ dần |
+| 7 | Conv Block | Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh (RGB), quan sát nhiều feature map ra từ nhiều kernel, histogram giá trị sau từng bước |
+| 8 | Full Pipeline | Xếp 2-3 Conv Block liên tiếp, forward 1 ảnh thật qua từng bước Conv / ReLU / Pool của từng block, theo dõi shape thu nhỏ dần và std activation co lại qua độ sâu |
+| 9 | BatchNorm | Tính μ, σ², x̂, γx̂+β bằng số trên 1 mini-batch; stack 10 lớp conv random có/không BN; train mode vs eval mode (running stats); BN giúp train nhanh hơn |
+| 10 | Softmax + Cross-Entropy | Kéo logits xem xác suất, loss `-log p` và gradient `p − y`; 1 ảnh chữ số đi hết mạng đã train tới tận loss |
+| 11 | Backprop & Training | Đường học, ảnh hưởng của learning rate, gradient norm từng lớp, kernel và feature map trước vs sau khi train |
+| 12 | Generalization | Overfit có chủ đích (ít data + nhãn sai), so sánh chồng dropout / weight decay / augmentation / BatchNorm / thêm data, early stopping, ảnh val bị đoán sai |
 
 ## Kiến trúc & nguyên tắc thiết kế
@@ -46,5 +50,6 @@ Codebase tách bạch rõ phần **tự viết để học** và phần **hạ t
 
 ```
-cnn_core/   toàn bộ phép toán CNN — tự cài đặt bằng vòng lặp NumPy thuần
+cnn_core/   toàn bộ phép toán CNN — phần 1 tự cài đặt bằng vòng lặp NumPy thuần,
+            phần 2 (layers.py, train.py) vectorized + có backward để train
 viz/        theme, helper vẽ matplotlib, widget Streamlit dùng chung
 pages/      1 file = 1 khái niệm, chỉ gọi vào cnn_core/ rồi vẽ kết quả
@@ -65,4 +70,5 @@ không bị lật 180°) — đúng quy ước dùng trong deep learning, khác
 | [Pandas](https://pandas.pydata.org/) | Backend cho `st.data_editor` — bảng số cho phép sửa tay từng ô input/kernel |
 | [Pillow](https://python-pillow.org/) | Đọc và resize ảnh người dùng upload ở trang Full Pipeline |
+| [scikit-learn](https://scikit-learn.org/) | Chỉ để lấy dataset `load_digits` (1797 ảnh chữ số 8×8, có sẵn offline) cho trang 9–12 |
 | [pytest](https://pytest.org/) | Test suite với giá trị kỳ vọng tính tay, dùng để tự chấm đúng/sai khi cài đặt `cnn_core/` |
 
@@ -110,5 +116,7 @@ CNN_Visualizer/
 │   ├── pooling.py                 # max_pool2d, avg_pool2d
 │   ├── activation.py              # relu, sigmoid
-│   └── block.py                   # ConvBlock — Conv → ReLU → Pool, make_random_kernels
+│   ├── block.py                   # ConvBlock — Conv → ReLU → Pool, make_random_kernels
+│   ├── layers.py                  # Conv2D, BatchNorm2D, Dropout, Dense... có forward + backward (vectorized)
+│   └── train.py                   # dataset digits, build_model, SGD momentum + weight decay, train()
 ├── viz/
 │   ├── theme.py                   # bảng màu, hero header, card, step-progress dots
@@ -172,14 +180,39 @@ element-wise trên toàn bộ feature map.
 điển: `conv2d_multichannel → relu → max_pool2d` (áp dụng riêng từng kênh vì
 `max_pool2d` chỉ nhận input 2D). Kernel được sinh ngẫu nhiên
-(`make_random_kernels`, phân phối `N(0, 0.1)`) — project này không cài đặt
-backprop/training, mục tiêu chỉ là quan sát forward pass, tương đương nhìn
-một model ngay sau khi khởi tạo.
+(`make_random_kernels`, phân phối `N(0, 0.1)`) — phần 1 không train, mục tiêu
+chỉ là quan sát forward pass, tương đương nhìn một model ngay sau khi khởi
+tạo (training nằm ở phần 2). `forward_steps` trả về kết quả sau từng
+bước để trang Full Pipeline đi qua được Conv / ReLU / Pool riêng lẻ.
+
+### Layers có backward (`cnn_core/layers.py`)
+
+Phần 2 cần train, nên vòng lặp tay quá chậm. Mỗi lớp (`Conv2D`, `ReLU`,
+`MaxPool2D`, `BatchNorm2D`, `Dropout`, `Flatten`, `Dense`) có `forward(x,
+train)` và `backward(dout)`, nhận batch `(N, H, W, C)`. Conv và pool xếp mọi
+cửa sổ `k×k` ra 1 mảng (`_windows`, tương đương im2col) rồi tính bằng 1 phép
+matmul/max; backward cộng dồn gradient ngược về từng pixel gốc
+(`_windows_backward`). `BatchNorm2D` dùng thống kê của batch khi train và
+running stats khi eval. `softmax_cross_entropy_backward` trả về `(p − y)/N`.
+
+Mọi backward được kiểm bằng gradient check (sai phân trung tâm), `Conv2D`
+được đối chiếu với `conv2d_multichannel` vòng lặp tay, và (nếu có `torch`)
+với `torch.nn.functional.conv2d` / `batch_norm`.
+
+### Training (`cnn_core/train.py`)
+
+Dataset `load_digits` (8×8, 10 lớp), 500 ảnh val cố định. Mô hình:
+`Conv(16) → [BN] → ReLU → Pool → Conv(32) → [BN] → ReLU → Pool → Dense(128→64)
+→ ReLU → [Dropout] → Dense(64→10)` — cố ý dư sức chứa để overfit hiện rõ.
+SGD momentum 0.9, weight decay chỉ áp cho `W`. Tuỳ chọn `label_noise` gán lại
+nhãn ngẫu nhiên cho 1 phần ảnh train để mô phỏng dữ liệu bẩn. Train full data
+~1 giây, val acc ~99%.
 
 ## Kiểm thử
 
-20 test case trong `tests/`, giá trị kỳ vọng được tính tay trên các ma trận
-nhỏ (4×4, 2×2) trước khi viết test — không dùng bất kỳ hàm nào từ
-`cnn_core/` để sinh ra "đáp án", tránh trường hợp test tự khớp với chính lỗi
-của cài đặt.
+Test phần 1 dùng giá trị kỳ vọng tính tay trên các ma trận nhỏ (4×4, 2×2) —
+không dùng hàm nào từ `cnn_core/` để sinh ra "đáp án", tránh trường hợp test
+tự khớp với chính lỗi của cài đặt. Test phần 2 dùng gradient check số học và
+đối chiếu với bản vòng lặp tay / torch. `tests/test_pages.py` chạy thử mọi
+trang Streamlit bằng `AppTest`.
 
 ```bash
@@ -189,11 +222,10 @@ pytest -v
 ## Trạng thái hoàn thành
 
-Toàn bộ 6 phase trong `TODO.md` đã hoàn tất — 20/20 test pass, cả 8 trang
-Streamlit chạy được từ input tuỳ chỉnh tới ảnh thật qua nhiều lớp Conv Block.
+Toàn bộ 6 phase trong `TODO.md` đã hoàn tất, cả 12 trang Streamlit chạy
+được — từ input tuỳ chỉnh, qua ảnh thật đi qua nhiều Conv Block, tới train
+thật và so sánh các kỹ thuật chống overfit.
 
 ## Định hướng mở rộng
 
-- So sánh output `conv2d` tự viết với `torch.nn.functional.conv2d` cùng
-  kernel — số phải khớp tuyệt đối (trong sai số dấu phẩy động)
 - Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur để xây trực giác
   "kernel = bộ dò 1 loại pattern"
```

```diff
diff --git a/TODO.md b/TODO.md
index 9159843..0cbb88f 100644
--- a/TODO.md
+++ b/TODO.md
@@ -104,6 +104,9 @@ CNN — pooling/block/pipeline chỉ là lắp ráp lại đúng 2 khối này.
 ## Ý mở rộng (sau khi xong hết, tùy hứng)
 
-- [ ] So sánh output `conv2d` tự viết với `torch.nn.functional.conv2d` cùng
+- [x] So sánh output `conv2d` tự viết với `torch.nn.functional.conv2d` cùng
       kernel — số phải khớp tuyệt đối (sai số float cho phép)
+      (`tests/test_layers.py::test_conv_and_batchnorm_match_torch`)
+- [x] Phần 2: BatchNorm, Softmax + Cross-Entropy, Backprop & Training,
+      Generalization (trang 9–12, `cnn_core/layers.py` + `train.py`)
 - [ ] Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur ở trang 1 để xây
       trực giác "kernel = bộ dò 1 loại pattern"
```

- [ ] **Step 3: Full verification**

Run: `pytest -q`
Expected: `56 passed`, possibly with 2 pre-existing `set_bad` deprecation warnings.

Run: `streamlit run app.py`. Open each page 7–12 in the browser. The home page must show 12 cards, all marked "Xong".

- [ ] **Step 4: Commit**

```bash
git add app.py README.md TODO.md
git commit -m "List pages 9-12 on home page and document training stack

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
