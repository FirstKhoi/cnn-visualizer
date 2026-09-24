# Why-CNN & What-It-Sees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add page 13 (Vì sao là CNN?) and page 14 (Mạng nhìn thấy gì?) on top of a new `cnn_core/analysis.py`, and pay off the 7 minors left by the previous final review.

**Architecture:** `cnn_core/train.py` gains an MLP arch, `count_params` and `shift_images`. `cnn_core/layers.py` gets a correct eval-mode BN backward, which saliency needs, and a race-free ReLU. Model-inspection tools live in `cnn_core/analysis.py`. The pages read trained models through the existing `cached_train`. Every gradient-based analysis runs on a `deepcopy`.

**Tech Stack:** numpy, Streamlit, matplotlib, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-24-why-cnn-what-it-sees-design.md`

**Provenance:** Every block below was prototyped and run in a scratch copy of the repo: 100/100 tests pass. Pages 13 and 14 render in about 7 s on first load, including training, and every caption was read back from a real AppTest run. Copy the blocks verbatim.

## Global Constraints

- numpy only; no new dependencies.
- No backward on a model that comes from `cached_train`: use `copy.deepcopy`, which `saliency_map` does internally.
- Every sentence of explanation must match numbers that page measures. Where a number depends on a user control, build the sentence from the measurement.
- UI copy in Vietnamese; palette and components from `viz/theme.py`; use only APIs that exist in Streamlit 1.38. AppTest's `switch_page` and `toast` exist in the installed version.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Saliency on the shared cached model.** The model must come back with identical params and the same `cache`/`grads` objects. Pinned by `test_saliency_matches_numeric_gradient_and_leaves_model_untouched`.
2. **BN in eval mode during backward.** It must use the constant-stats formula. Pinned by `test_batchnorm_eval_mode_gradient`, and indirectly by the saliency test, whose model has BN.
3. **Leaving page 12 and coming back.** Controls must survive. Pinned by `test_generalization_page_keeps_controls_after_visiting_another_page`, which uses real `switch_page`.
4. **Page 14 at extreme settings** (kernel 5 × 4 blocks, RF 76 px on a 96 px input) must neither crash nor crop wrongly. Pinned by the Task 7 interaction step.
5. **Page 13 with BN off.** The CNN can lose to the MLP on clean images at 50 samples (−1.6%). The caption must print the signed gap and never claim CNN "wins". Pinned by the Task 6 interaction step.

---

### Task 1: Layers — race-free ReLU, eval-mode BN backward

**Files:** Modify `cnn_core/layers.py`, `tests/test_layers.py`

**Interfaces:**
- Produces: `BatchNorm2D.backward` is correct after `forward(train=False)`; `ReLU.forward` never reads `self.mask`.

- [ ] **Step 1: Failing tests** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/tests/test_layers.py b/tests/test_layers.py
index a06d96b..a84eb3b 100644
--- a/tests/test_layers.py
+++ b/tests/test_layers.py
@@ -148,2 +148,42 @@ def test_conv_and_batchnorm_match_torch():
     ).permute(0, 2, 3, 1)
     np.testing.assert_allclose(bn_ours, bn_theirs.numpy(), atol=1e-8)
+
+
+def test_flatten_gradients():
+    from cnn_core.layers import Flatten
+
+    rng = np.random.default_rng(9)
+    check_layer_grads(Flatten(), rng.normal(size=(2, 3, 3, 2)), rng)
+
+
+@pytest.mark.parametrize("shape", [(4, 3, 3, 3), (6, 3)])
+def test_batchnorm_eval_mode_gradient(shape):
+    """Eval dùng running stats (hằng số) -> gradient khác hẳn công thức train."""
+    rng = np.random.default_rng(10)
+    bn = BatchNorm2D(3)
+    bn.params["gamma"] = rng.normal(size=3)
+    bn.running_mean = rng.normal(size=3)
+    bn.running_var = rng.uniform(0.5, 2.0, size=3)
+    x = rng.normal(size=shape)
+    g = rng.normal(size=shape)
+    bn.forward(x, train=False)
+    dx = bn.backward(g)
+
+    def loss():
+        return float(np.sum(bn.forward(x, train=False) * g))
+
+    np.testing.assert_allclose(dx, numeric_grad(loss, x), atol=1e-6)
+
+
+def test_relu_forward_does_not_read_mask_back_from_attribute():
+    """2 phiên dùng chung 1 model đã cache: phiên khác có thể ghi đè self.mask
+    giữa lúc gán và lúc dùng. forward phải tính từ biến cục bộ."""
+
+    class Racing(ReLU):
+        def __getattribute__(self, name):
+            if name == "mask":
+                return np.zeros((1,), dtype=bool)  # "mask của phiên khác"
+            return super().__getattribute__(name)
+
+    x = np.array([[-1.0, 2.0, 3.0]])
+    np.testing.assert_array_equal(Racing().forward(x), [[0.0, 2.0, 3.0]])
```

- [ ] **Step 2: Run** `pytest tests/test_layers.py -q`
Expected: 2 BN-eval failures and 1 ReLU failure. The Flatten check passes, since it pins behaviour that is already correct.

- [ ] **Step 3: Implement** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/cnn_core/layers.py b/cnn_core/layers.py
index ecf31e8..d2c3270 100644
--- a/cnn_core/layers.py
+++ b/cnn_core/layers.py
@@ -87,6 +87,9 @@ class Conv2D(Layer):
 class ReLU(Layer):
     def forward(self, x, train=True):
-        self.mask = x > 0
-        return x * self.mask
+        # tính vào biến cục bộ: không đọc lại self.mask, để 2 phiên dùng chung 1
+        # model đã cache (st.cache_resource) không giẫm lên mask của nhau
+        mask = x > 0
+        self.mask = mask
+        return x * mask
 
     def backward(self, dout):
@@ -122,4 +125,5 @@ class BatchNorm2D(Layer):
     train=True: dùng mean/var của CHÍNH batch này + cập nhật running stats.
     train=False: dùng running stats (1 ảnh lẻ lúc predict không có "batch").
+    backward() dùng đúng công thức của mode ở lần forward gần nhất.
     """
 
@@ -142,12 +146,16 @@ class BatchNorm2D(Layer):
         std = np.sqrt(var + self.eps)
         x_hat = (x - mean) / std
-        self.cache = (x_hat, std, axes)
+        self.cache = (x_hat, std, axes, train)
         return self.params["gamma"] * x_hat + self.params["beta"]
 
     def backward(self, dout):
-        x_hat, std, axes = self.cache
-        M = dout.size // dout.shape[-1]  # số phần tử mỗi kênh
+        x_hat, std, axes, train = self.cache
         self.grads = {"gamma": (dout * x_hat).sum(axis=axes), "beta": dout.sum(axis=axes)}
         dx_hat = dout * self.params["gamma"]
+        if not train:
+            # eval: mean/std là hằng số (running stats) -> BN chỉ là phép affine
+            return dx_hat / std
+        M = dout.size // dout.shape[-1]  # số phần tử mỗi kênh
+        # train: mean/var phụ thuộc cả batch nên gradient có thêm 2 số hạng
         return (M * dx_hat - dx_hat.sum(axis=axes) - x_hat * (dx_hat * x_hat).sum(axis=axes)) / (M * std)
 
```

- [ ] **Step 4: Run** `pytest tests/test_layers.py -q` → `18 passed`; `pytest -q` → `74 passed`

- [ ] **Step 5: Commit**

```bash
git add cnn_core/layers.py tests/test_layers.py
git commit -m "Fix eval-mode BatchNorm backward and make ReLU safe on shared models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Train — MLP arch, `count_params`, `shift_images`

**Files:** Modify `cnn_core/train.py`, `tests/test_train.py`

**Interfaces:**
- Produces: `TrainConfig.arch = "cnn"` (a new first field); `build_model` supports `"mlp"` and raises `ValueError` on unknown values; `count_params(model) -> int`; `shift_images(x, dy, dx) -> ndarray`. Parameter counts without BN: CNN 13 706, MLP 24 090.

- [ ] **Step 1: Failing tests** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/tests/test_train.py b/tests/test_train.py
index dcc3ae3..f6ed6b9 100644
--- a/tests/test_train.py
+++ b/tests/test_train.py
@@ -82,2 +82,28 @@ def test_load_digits_split_rejects_bad_label_noise(bad):
     with pytest.raises(ValueError):
         load_digits_split(100, label_noise=bad)
+
+
+def test_mlp_arch_builds_and_learns():
+    from cnn_core.train import count_params
+
+    cnn, mlp = build_model(TrainConfig(use_bn=False)), build_model(TrainConfig(arch="mlp", use_bn=False))
+    assert count_params(cnn) == 13706
+    assert count_params(mlp) == 24090  # MLP được cho NHIỀU tham số hơn CNN
+    assert param_layer_names(mlp) == ["Dense 1", "Dense 2", "Dense 3"]
+    _, history = train(TrainConfig(arch="mlp", epochs=3))
+    assert history["val_acc"][-1] > 0.9
+
+
+def test_unknown_arch_is_rejected():
+    with pytest.raises(ValueError):
+        build_model(TrainConfig(arch="resnet"))
+
+
+def test_shift_images_moves_every_image_by_exact_offset():
+    from cnn_core.train import shift_images
+
+    x = np.arange(32.0).reshape(2, 4, 4, 1)
+    out = shift_images(x, 1, -1)
+    np.testing.assert_array_equal(out[:, 1:, :3], x[:, :3, 1:])
+    assert np.all(out[:, 0] == 0) and np.all(out[:, :, 3] == 0)
+    np.testing.assert_array_equal(shift_images(x, 0, 0), x)
```

- [ ] **Step 2: Run** `pytest tests/test_train.py -q` → 3 failures: `ImportError` on `count_params`/`shift_images`, and the arch test does not raise.

- [ ] **Step 3: Implement** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/cnn_core/train.py b/cnn_core/train.py
index eddec00..0eaad00 100644
--- a/cnn_core/train.py
+++ b/cnn_core/train.py
@@ -34,4 +34,5 @@ MAX_TRAIN_SIZE = 1797 - VAL_SIZE
 @dataclass(frozen=True)
 class TrainConfig:
+    arch: str = "cnn"  # "cnn" | "mlp" (trang 13 so sánh)
     use_bn: bool = True
     dropout: float = 0.0
@@ -83,5 +84,13 @@ def build_model(cfg: TrainConfig) -> list[Layer]:
 
     Cố ý dư sức chứa (~15k tham số cho vài trăm ảnh) để overfit hiện rõ ở trang 12.
+
+    arch="mlp": Flatten -> Dense(64 -> 160) -> [BN] -> ReLU -> Dense(160 -> 80) -> [BN]
+    -> ReLU -> [Dropout] -> Dense(80 -> 10) — ~24k tham số, NHIỀU hơn CNN, không
+    có locality hay weight sharing.
     """
+    if cfg.arch == "mlp":
+        return _build_mlp(cfg)
+    if cfg.arch != "cnn":
+        raise ValueError(f"arch phải là 'cnn' hoặc 'mlp', nhận {cfg.arch!r}")
     rng = np.random.default_rng(cfg.seed)
     model: list[Layer] = [Conv2D(1, 16, 3, padding=1, rng=rng)]
@@ -98,4 +107,24 @@ def build_model(cfg: TrainConfig) -> list[Layer]:
 
 
+def _build_mlp(cfg: TrainConfig) -> list[Layer]:
+    rng = np.random.default_rng(cfg.seed)
+    model: list[Layer] = [Flatten(), Dense(8 * 8, 160, rng=rng)]
+    if cfg.use_bn:
+        model.append(BatchNorm2D(160))
+    model += [ReLU(), Dense(160, 80, rng=rng)]
+    if cfg.use_bn:
+        model.append(BatchNorm2D(80))
+    model.append(ReLU())
+    if cfg.dropout > 0:
+        model.append(Dropout(cfg.dropout, rng=rng))
+    model.append(Dense(80, 10, rng=rng))
+    return model
+
+
+def count_params(model: list[Layer]) -> int:
+    """Tổng số tham số học được (W, b, gamma, beta) của mọi lớp."""
+    return sum(p.size for layer in model for p in layer.params.values())
+
+
 def forward(model: list[Layer], x: np.ndarray, train: bool = False) -> np.ndarray:
     for layer in model:
@@ -132,4 +161,15 @@ def augment_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
 
 
+def shift_images(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
+    """Dịch CẢ batch (N, H, W, C) đúng dy pixel xuống, dx pixel sang phải (số âm =
+    lên/trái). Phần lộ ra điền 0, phần trượt ra ngoài bị cắt."""
+    _, H, W, _ = x.shape
+    out = np.zeros_like(x)
+    out[:, max(dy, 0) : H + min(dy, 0), max(dx, 0) : W + min(dx, 0)] = x[
+        :, max(-dy, 0) : H + min(-dy, 0), max(-dx, 0) : W + min(-dx, 0)
+    ]
+    return out
+
+
 def param_layer_names(model: list[Layer]) -> list[str]:
     """Tên các lớp có weight "W" (conv/dense), đánh số theo thứ tự: ["Conv2D 1", "Conv2D 2", "Dense 1"]."""
```

- [ ] **Step 4: Run** `pytest tests/test_train.py -q` → `14 passed`; `pytest -q` → `77 passed`

- [ ] **Step 5: Commit**

```bash
git add cnn_core/train.py tests/test_train.py
git commit -m "Add MLP baseline, parameter count and exact image shift to training module

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `cnn_core/analysis.py`

**Files:** Create `cnn_core/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `Layer`, `softmax`; `Conv2D`, `ReLU`, `MaxPool2D`; `TrainConfig`, `train`, `build_model`, `load_digits_split`.
- Produces:
  - `receptive_field(specs) -> int`
  - `receptive_field_map(layers, x) -> (H, W)`
  - `center_concentration(grad, r) -> (share, area, edge)`
  - `pca_2d(features) -> (N, 2)`
  - `nearest_centroid_accuracy(f_tr, y_tr, f_val, y_val) -> float`
  - `occlusion_map(model, image, label, patch=2) -> (H-patch+1, W-patch+1)`
  - `saliency_map(model, image, label) -> (H, W)`
  - `confusion_matrix(y_true, y_pred, n_classes=10)`

- [ ] **Step 1: Failing tests** — create `tests/test_analysis.py`:

```python
import numpy as np
import pytest

from cnn_core.analysis import (
    center_concentration,
    confusion_matrix,
    nearest_centroid_accuracy,
    occlusion_map,
    pca_2d,
    receptive_field,
    receptive_field_map,
    saliency_map,
)
from cnn_core.layers import Conv2D, MaxPool2D, ReLU
from cnn_core.train import TrainConfig, build_model, load_digits_split, train

BLOCK = [(3, 1), (2, 2)]  # Conv3×3 stride 1, rồi MaxPool 2×2 stride 2


@pytest.mark.parametrize(
    "specs,expected",
    [
        ([(3, 1)], 3),
        (BLOCK, 4),
        (BLOCK * 2, 10),
        (BLOCK * 3, 22),
        ([(3, 1)] * 3, 7),  # 3 conv 3×3 liên tiếp = 1 conv 7×7 về vùng nhìn
        ([(5, 2), (3, 1)], 9),  # stride 2 làm bước của lớp sau dài gấp đôi
    ],
)
def test_receptive_field_formula(specs, expected):
    assert receptive_field(specs) == expected


@pytest.mark.parametrize("blocks,expected", [(1, 4), (2, 10), (3, 22)])
def test_receptive_field_map_matches_formula(blocks, expected):
    rng = np.random.default_rng(0)
    layers, ch = [], 1
    for _ in range(blocks):
        layers += [Conv2D(ch, 4, 3, padding=1, rng=rng), ReLU(), MaxPool2D(2, 2)]
        ch = 4
    grad = receptive_field_map(layers, rng.normal(size=(16, 48, 48, 1)))
    rows, cols = np.nonzero(grad > 0)
    assert rows.max() - rows.min() + 1 == expected
    assert cols.max() - cols.min() + 1 == expected


def test_pca_2d_first_axis_follows_largest_variance():
    rng = np.random.default_rng(1)
    f = np.column_stack([rng.normal(0, 10, 200), rng.normal(0, 1, 200), rng.normal(0, 0.1, 200)])
    z = pca_2d(f)
    assert z.shape == (200, 2)
    assert abs(np.corrcoef(z[:, 0], f[:, 0])[0, 1]) > 0.99
    assert z[:, 0].var() > z[:, 1].var()


def test_nearest_centroid_accuracy_on_separated_and_mixed_clusters():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 30)
    separated = np.eye(3)[y] * 10 + rng.normal(0, 0.1, (90, 3))
    assert nearest_centroid_accuracy(separated, y, separated, y) == 1.0
    noise = rng.normal(size=(90, 3))
    assert nearest_centroid_accuracy(noise, y, rng.normal(size=(90, 3)), y) < 0.6


@pytest.fixture(scope="module")
def trained():
    model, _ = train(TrainConfig(epochs=5))
    _, _, x_val, y_val = load_digits_split()
    return model, x_val, y_val


def test_occlusion_map_blank_corner_does_not_matter(trained):
    model, x_val, y_val = trained
    image = x_val[0].copy()
    image[:2, :2] = 0.0  # góc trên-trái vốn trống -> che nó không đổi gì
    heat = occlusion_map(model, image, int(y_val[0]), patch=2)
    assert heat.shape == (7, 7)
    assert heat[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert heat.max() > 0.05  # có ít nhất 1 vùng mà che đi là mạng mất tự tin


def test_saliency_matches_numeric_gradient_and_leaves_model_untouched(trained):
    model, x_val, y_val = trained
    image, label = x_val[1], int(y_val[1])
    before = [{k: v.copy() for k, v in layer.params.items()} for layer in model]
    state = [(layer.__dict__.get("cache"), layer.grads) for layer in model]

    sal = saliency_map(model, image, label)

    # model gốc (dùng chung trong cache) không bị ghi cache/grads mới
    for layer, (cache, grads) in zip(model, state):
        assert layer.__dict__.get("cache") is cache
        assert layer.grads is grads

    def logit(img):
        x = img[None]
        for layer in model:
            x = layer.forward(x, train=False)
        return x[0, label]

    i, j = np.unravel_index(sal.argmax(), sal.shape)
    bumped, dropped = image.copy(), image.copy()
    bumped[i, j, 0] += 1e-5
    dropped[i, j, 0] -= 1e-5
    numeric = abs(logit(bumped) - logit(dropped)) / 2e-5
    assert sal[i, j] == pytest.approx(numeric, rel=1e-3)
    for layer, params in zip(model, before):
        for k, v in params.items():
            np.testing.assert_array_equal(layer.params[k], v)


def test_confusion_matrix_counts():
    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), n_classes=3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 1, 0]])


def test_random_network_representation_is_less_separable_than_trained(trained):
    model, x_val, y_val = trained
    x_tr, y_tr, _, _ = load_digits_split(500)

    def logits(m, x):
        for layer in m:
            x = layer.forward(x, train=False)
        return x

    random_model = build_model(TrainConfig())
    acc_trained = nearest_centroid_accuracy(logits(model, x_tr), y_tr, logits(model, x_val), y_val)
    acc_random = nearest_centroid_accuracy(logits(random_model, x_tr), y_tr, logits(random_model, x_val), y_val)
    assert acc_trained > 0.95 > 0.8 > acc_random


def test_center_concentration_uniform_vs_peaked():
    uniform = np.zeros((20, 20))
    uniform[4:16, 4:16] = 1.0  # receptive field 12×12, ảnh hưởng đều
    share, area, edge = center_concentration(uniform, 12)
    assert share == pytest.approx(area)
    assert edge == pytest.approx(1.0)

    yy, xx = np.mgrid[:20, :20]
    peaked = np.exp(-((yy - 9.5) ** 2 + (xx - 9.5) ** 2) / 8.0) * (uniform > 0)
    share, area, edge = center_concentration(peaked, 12)
    assert share > 2 * area
    assert edge < 0.1
```

- [ ] **Step 2: Run** `pytest tests/test_analysis.py -q` → `ModuleNotFoundError: No module named 'cnn_core.analysis'`

- [ ] **Step 3: Implement** — create `cnn_core/analysis.py`:

```python
"""Công cụ "soi" mạng: receptive field, biểu diễn qua từng lớp, pixel nào quyết
định dự đoán. Dùng cho trang 13–14.

Mọi hàm cần backward (saliency) chạy trên BẢN SAO model — model trong
st.cache_resource là dùng chung giữa các phiên, không được ghi cache/grads lên đó.
"""

from __future__ import annotations

import copy

import numpy as np

from cnn_core.layers import Layer, softmax


def receptive_field(specs: list[tuple[int, int]]) -> int:
    """Receptive field lý thuyết (px, theo 1 chiều) của 1 neuron ở lớp cuối.

    specs: [(kernel, stride), ...] theo thứ tự từ input đi lên, gồm cả pool
    (vd Conv3×3 stride 1 = (3, 1), MaxPool 2×2 stride 2 = (2, 2)).

    Mỗi lớp nới vùng nhìn thêm (k − 1) bước, mỗi bước dài `jump` pixel input;
    stride > 1 làm các bước của MỌI lớp phía sau dài ra gấp stride lần.
    """
    r, jump = 1, 1
    for k, stride in specs:
        r += (k - 1) * jump
        jump *= stride
    return r


def receptive_field_map(layers: list[Layer], x: np.ndarray) -> np.ndarray:
    """Đo receptive field bằng gradient: trung bình |∂ neuron giữa / ∂ input|.

    x: batch (N, H, W, C). Chạy forward qua `layers`, đặt gradient = 1 tại vị
    trí chính giữa output (mọi kênh), backward về input. Pixel nào nằm ngoài
    receptive field sẽ có gradient đúng bằng 0.

    Returns: mảng (H, W) — trung bình |gradient| theo batch và kênh.
    """
    out = x
    for layer in layers:
        out = layer.forward(out, train=False)
    dout = np.zeros_like(out)
    dout[:, out.shape[1] // 2, out.shape[2] // 2, :] = 1.0
    for layer in reversed(layers):
        dout = layer.backward(dout)
    return np.abs(dout).mean(axis=(0, 3))


def center_concentration(grad: np.ndarray, r: int) -> tuple[float, float, float]:
    """Receptive field có "đều" không? Trong hộp r×r (bắt đầu từ ô gradient ≠ 0
    đầu tiên), đo:

    - share: phần ảnh hưởng nằm trong hình vuông giữa, cạnh ≈ r/2;
    - area: diện tích hình vuông đó / diện tích hộp (ảnh hưởng đều thì share = area);
    - edge: gradient ở 2 mép hộp so với tâm, trên hàng qua tâm (1 = đều, ~0 = mép vô nghĩa).
    """
    rows, cols = np.nonzero(grad > 0)
    box = grad[rows.min() : rows.min() + r, cols.min() : cols.min() + r]
    c, h = r // 2, max(r // 4, 1)
    core = box[c - h : c + h + r % 2, c - h : c + h + r % 2]
    profile = box[c] / box[c].max()
    return float(core.sum() / box.sum()), core.size / box.size, float((profile[0] + profile[-1]) / 2)


def pca_2d(features: np.ndarray) -> np.ndarray:
    """Chiếu (N, D) xuống 2 chiều có phương sai lớn nhất (PCA qua SVD)."""
    f = features.reshape(len(features), -1)
    f = f - f.mean(axis=0)
    _, _, vt = np.linalg.svd(f, full_matrices=False)
    return f @ vt[:2].T


def nearest_centroid_accuracy(
    f_train: np.ndarray, y_train: np.ndarray, f_val: np.ndarray, y_val: np.ndarray
) -> float:
    """Độ "tách lớp" của 1 biểu diễn: gán mỗi ảnh val cho lớp có tâm (trung bình
    ảnh train của lớp đó) gần nhất. Cao = các lớp nằm thành cụm riêng biệt."""
    f_train = f_train.reshape(len(f_train), -1)
    f_val = f_val.reshape(len(f_val), -1)
    classes = np.unique(y_train)
    centroids = np.stack([f_train[y_train == c].mean(axis=0) for c in classes])
    dist = ((f_val[:, None, :] - centroids[None]) ** 2).sum(axis=-1)
    return float(np.mean(classes[dist.argmin(axis=1)] == y_val))


def _predict_probs(model: list[Layer], x: np.ndarray) -> np.ndarray:
    for layer in model:
        x = layer.forward(x, train=False)
    return softmax(x)


def occlusion_map(model: list[Layer], image: np.ndarray, label: int, patch: int = 2) -> np.ndarray:
    """Che lần lượt từng ô patch×patch bằng 0, đo xác suất lớp đúng rơi bao nhiêu.

    image: (H, W, C). Returns (H − patch + 1, W − patch + 1): giá trị lớn = vùng
    mà thiếu nó thì mạng mất tự tin nhất. Chỉ dùng forward nên an toàn trên
    model dùng chung (cache của từng lớp bị ghi đè, nhưng trọng số không đổi).
    """
    H, W, _ = image.shape
    positions = [(i, j) for i in range(H - patch + 1) for j in range(W - patch + 1)]
    batch = np.repeat(image[None], len(positions) + 1, axis=0)  # phần tử cuối = ảnh gốc
    for n, (i, j) in enumerate(positions):
        batch[n, i : i + patch, j : j + patch, :] = 0.0
    probs = _predict_probs(model, batch)[:, label]
    return (probs[-1] - probs[:-1]).reshape(H - patch + 1, W - patch + 1)


def saliency_map(model: list[Layer], image: np.ndarray, label: int) -> np.ndarray:
    """|∂ logit lớp đúng / ∂ pixel| — pixel nào đổi 1 chút là logit đổi nhiều.

    image: (H, W, C). Chạy trên deepcopy(model) để không ghi cache/grads lên
    model gốc. Returns (H, W): max |gradient| theo kênh.
    """
    model = copy.deepcopy(model)
    x = image[None]
    for layer in model:
        x = layer.forward(x, train=False)
    dout = np.zeros_like(x)
    dout[0, label] = 1.0
    for layer in reversed(model):
        dout = layer.backward(dout)
    return np.abs(dout[0]).max(axis=-1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """cm[i, j] = số ảnh có nhãn đúng i mà mạng đoán là j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm
```

- [ ] **Step 4: Run** `pytest tests/test_analysis.py -q` → `16 passed`; `pytest -q` → `93 passed`

- [ ] **Step 5: Commit**

```bash
git add cnn_core/analysis.py tests/test_analysis.py
git commit -m "Add model analysis tools: receptive field, PCA, occlusion, saliency

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Viz — automatic colormap, scatter grid, colours for pages 13–14

**Files:** Modify `viz/matrix_view.py`, `viz/theme.py`, `pages/10_Softmax_Cross_Entropy.py`, `tests/test_viz.py`

**Interfaces:**
- Produces:
  - `fig_feature_maps(..., cmap=None)`: `None` picks `DIV` when the data has negatives, otherwise `WARM`.
  - `CLASS_COLORS`
  - `fig_scatter_grid(panels, labels, titles, rows=1, point_size=6)`
  - `PAGE_COLORS[13]`, `PAGE_COLORS[14]`

  Page 10 now uses the automatic colormap, and its copy reads 0.095.

- [ ] **Step 1: Failing tests** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/tests/test_viz.py b/tests/test_viz.py
index bbd0b70..c7d1c6a 100644
--- a/tests/test_viz.py
+++ b/tests/test_viz.py
@@ -29,2 +29,23 @@ def test_shared_scale_is_symmetric_around_zero_when_negative():
     fig = fig_feature_maps(np.array([[[-3.0, 1.0]]]), shared_scale=True)
     assert fig.axes[0].images[0].get_clim() == (-3.0, 3.0)
+
+
+def test_feature_maps_pick_diverging_colormap_only_for_signed_data():
+    from viz.matrix_view import WARM, fig_feature_maps
+
+    signed = fig_feature_maps(np.array([[[-1.0, 2.0]]]), shared_scale=True)
+    positive = fig_feature_maps(np.array([[[0.0, 2.0]]]), shared_scale=True)
+    assert signed.axes[0].images[0].get_cmap().name == DIV.name
+    assert positive.axes[0].images[0].get_cmap().name == WARM.name
+
+
+def test_scatter_grid_draws_every_panel_with_class_colors():
+    from viz.matrix_view import fig_scatter_grid
+
+    labels = np.array([0, 1, 2, 0])
+    panels = [np.random.default_rng(i).normal(size=(4, 2)) for i in range(5)]
+    fig = fig_scatter_grid(panels, labels, [f"p{i}" for i in range(5)], rows=2)
+    drawn = [ax for ax in fig.axes if ax.collections]
+    assert len(drawn) == 5
+    assert all(len(ax.collections) == 3 for ax in drawn)  # 1 nhóm điểm mỗi lớp
+    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["0", "1", "2"]
```

- [ ] **Step 2: Run** `pytest tests/test_viz.py -q` → 2 failures: the cmap is not DIV, and `ImportError` on `fig_scatter_grid`.

- [ ] **Step 3: Implement** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/viz/matrix_view.py b/viz/matrix_view.py
index 276bb9f..3c1aca7 100644
--- a/viz/matrix_view.py
+++ b/viz/matrix_view.py
@@ -149,5 +149,5 @@ def fig_feature_maps(
     titles: list[str] | None = None,
     max_cols: int = 4,
-    cmap=WARM,
+    cmap=None,
     shared_scale: bool = False,
 ) -> plt.Figure:
@@ -162,6 +162,10 @@ def fig_feature_maps(
         lớn giữa các kênh (thấy kênh nào "chết", kênh nào trội). Nếu có số
         âm, thang đối xứng quanh 0.
+    cmap=None: tự chọn DIV (phân kỳ, 0 = trắng) nếu dữ liệu có số âm, WARM
+        nếu không — để giá trị 0 không rơi vào giữa 1 colormap tuần tự.
     """
     feature_maps = np.asarray(feature_maps)
+    if cmap is None:
+        cmap = DIV if feature_maps.min() < 0 else WARM
     channels = feature_maps.shape[-1]
     cols = min(max_cols, channels)
@@ -299,2 +303,37 @@ def fig_bars(
     fig.tight_layout()
     return fig
+
+
+CLASS_COLORS = plt.get_cmap("tab10").colors  # 10 màu phân biệt cho 10 chữ số
+
+
+def fig_scatter_grid(
+    panels: list[np.ndarray],
+    labels: np.ndarray,
+    titles: list[str],
+    rows: int = 1,
+    point_size: float = 6,
+) -> plt.Figure:
+    """Lưới scatter 2D (vd PCA của từng lớp), mỗi điểm tô màu theo nhãn lớp.
+
+    panels: list mảng (N, 2), cùng thứ tự điểm với `labels`. Có 1 legend chung.
+    """
+    cols = int(np.ceil(len(panels) / rows))
+    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
+    axes = axes.ravel()
+    classes = np.unique(labels)
+    for ax, points, title in zip(axes, panels, titles):
+        for c in classes:
+            sel = labels == c
+            ax.scatter(points[sel, 0], points[sel, 1], s=point_size, color=CLASS_COLORS[int(c) % 10], alpha=0.75, linewidths=0)
+        ax.set_title(title, fontsize=9)
+        ax.set_xticks([])
+        ax.set_yticks([])
+        for spine in ax.spines.values():
+            spine.set_color(BORDER)
+    for ax in axes[len(panels) :]:
+        ax.axis("off")
+    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[int(c) % 10], label=str(c)) for c in classes]
+    fig.legend(handles=handles, loc="lower center", ncol=len(classes), frameon=False, fontsize=8)
+    fig.tight_layout(rect=(0, 0.06, 1, 1))
+    return fig
```

```diff
diff --git a/viz/theme.py b/viz/theme.py
index bc588e0..684d424 100644
--- a/viz/theme.py
+++ b/viz/theme.py
@@ -36,4 +36,6 @@ PAGE_COLORS = {
     11: "#E84393",  # Backprop & Training — hồng đậm
     12: "#8E44AD",  # Generalization — tím đậm
+    13: "#16A085",  # Vì sao là CNN — xanh lục đậm
+    14: "#D35400",  # Mạng nhìn thấy gì — cam cháy
 }
 
```

```diff
diff --git a/pages/10_Softmax_Cross_Entropy.py b/pages/10_Softmax_Cross_Entropy.py
index 463ff22..7c809f7 100644
--- a/pages/10_Softmax_Cross_Entropy.py
+++ b/pages/10_Softmax_Cross_Entropy.py
@@ -5,5 +5,5 @@ import streamlit as st
 from cnn_core.layers import cross_entropy, softmax, softmax_cross_entropy_backward
 from cnn_core.train import TrainConfig, forward_trace, load_digits_split
-from viz.matrix_view import DIV, fig_bars, fig_feature_maps
+from viz.matrix_view import fig_bars, fig_feature_maps
 from viz.theme import ACCENT, BORDER, PAGE_COLORS, TEXT, callout, formula_box, hero, inject_base_css
 from viz.widgets import cached_train, step_controls
@@ -73,5 +73,5 @@ with st.container(border=True):
         st.markdown(
             "- Đường cong **dốc đứng khi p → 0**: đoán sai mà còn tự tin bị phạt cực nặng "
-            "(p = 0.01 → loss 4.6), trong khi từ 0.9 lên 0.99 chỉ bớt được 0.09.\n"
+            "(p = 0.01 → loss 4.6), trong khi từ 0.9 lên 0.99 chỉ bớt được 0.095.\n"
             "- Gradient **p − y** gọn đến bất ngờ: logit lớp đúng bị đẩy *lên* một lượng "
             "(1 − p), mọi logit khác bị đẩy *xuống* đúng bằng xác suất của nó.\n"
@@ -106,5 +106,5 @@ with st.container(border=True):
         st.pyplot(fig)
     elif out.ndim == 4:
-        st.pyplot(fig_feature_maps(out[0], max_cols=8, shared_scale=True, cmap=DIV if name == "Conv2D" else "magma"))
+        st.pyplot(fig_feature_maps(out[0], max_cols=8, shared_scale=True))
     elif name == "Cross-Entropy":
         formula_box(f"L = −log p{label} = −log({probs[0, label]:.4f}) = {loss:.4f}", color=COLOR)
```

- [ ] **Step 4: Run** `pytest tests/test_viz.py -q` → `5 passed`; `pytest -q` → `95 passed`

- [ ] **Step 5: Commit**

```bash
git add viz/matrix_view.py viz/theme.py pages/10_Softmax_Cross_Entropy.py tests/test_viz.py
git commit -m "Pick diverging colormap for signed maps and add class scatter grid

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Page 12 — toast, state survives leaving the page, float rounding, all presets tested

**Files:** Modify `pages/12_Generalization.py`, `tests/test_pages.py`

- [ ] **Step 1: Failing tests** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/tests/test_pages.py b/tests/test_pages.py
index 768e2bf..f009563 100644
--- a/tests/test_pages.py
+++ b/tests/test_pages.py
@@ -94,2 +94,34 @@ def test_training_page_weight_change_copy_follows_the_measurement():
     assert "gần như giống hệt" not in captions
     assert "na ná" not in captions
+
+
+def test_generalization_page_toasts_on_duplicate_run():
+    at = AppTest.from_file(str(ROOT / "pages" / "12_Generalization.py"), default_timeout=120).run()
+    at.button[0].click().run()  # preset 1 đã có sẵn từ lần mở đầu
+    assert any("đã có trong so sánh" in t.value for t in at.toast)
+    assert len(at.session_state["g_runs"]) == 1
+
+
+def test_generalization_page_keeps_controls_after_visiting_another_page():
+    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
+    at.switch_page("pages/12_Generalization.py").run()
+    at.selectbox(key="g_preset").set_value("2. + Dropout 0.5").run()
+    at.switch_page("pages/6_Activation.py").run()  # Streamlit dọn state widget trang 12
+    at.switch_page("pages/12_Generalization.py").run()
+    assert not at.exception
+    assert at.session_state["g_preset"] == "2. + Dropout 0.5"
+    assert at.session_state["g_dropout"] == 0.5
+
+
+def test_generalization_presets_set_every_control():
+    source = (ROOT / "pages" / "12_Generalization.py").read_text()
+    presets_src = source[source.index("OVERFIT = dict(") : source.index("def apply_preset")]
+    namespace = {"MAX_TRAIN_SIZE": 1297}
+    exec(presets_src, namespace)
+
+    at = AppTest.from_file(str(ROOT / "pages" / "12_Generalization.py"), default_timeout=120).run()
+    for name, values in namespace["PRESETS"].items():
+        at.selectbox(key="g_preset").set_value(name).run()
+        assert not at.exception
+        for key, value in values.items():
+            assert at.session_state[f"g_{key}"] == value, (name, key)
```

- [ ] **Step 2: Run** `pytest tests/test_pages.py -q -k "toasts or keeps_controls or every_control"`. Expected: `toasts` and `keeps_controls` FAIL (after switching back, `g_preset` resets to preset 1); `every_control` passes, since it pins existing behaviour.

- [ ] **Step 3: Implement** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/pages/12_Generalization.py b/pages/12_Generalization.py
index 37b1c70..844461a 100644
--- a/pages/12_Generalization.py
+++ b/pages/12_Generalization.py
@@ -59,4 +59,11 @@ def run_label(cfg: TrainConfig) -> str:
 
 
+WIDGET_KEYS = ["g_preset"] + [f"g_{k}" for k in OVERFIT]
+
+# Streamlit xoá state của widget khi rời trang (widget không còn được vẽ). Bản
+# sao "_g_*" không phải widget nên còn nguyên -> dùng nó khôi phục khi quay lại.
+for key in WIDGET_KEYS:
+    if key not in st.session_state and f"_{key}" in st.session_state:
+        st.session_state[key] = st.session_state[f"_{key}"]
 if "g_train_size" not in st.session_state:
     st.session_state["g_preset"] = next(iter(PRESETS))
@@ -76,8 +83,16 @@ with st.container(border=True):
     c3.slider("Số epoch", 10, 100, key="g_epochs", step=10)
 
-    cfg = TrainConfig(**{k: st.session_state[f"g_{k}"] for k in OVERFIT})
+    for key in WIDGET_KEYS:
+        st.session_state[f"_{key}"] = st.session_state[key]
+
+    # làm tròn: slider float có thể trả 0.30000000000000004 -> config "khác" 0.3
+    cfg = TrainConfig(
+        **{k: round(v, 4) if isinstance(v, float) else v for k, v in ((k, st.session_state[f"g_{k}"]) for k in OVERFIT)}
+    )
     b1, b2, _ = st.columns([1, 1, 3])
     if b1.button("Chạy & thêm vào so sánh", type="primary", use_container_width=True):
-        if cfg not in st.session_state["g_runs"]:
+        if cfg in st.session_state["g_runs"]:
+            st.toast("Config này đã có trong so sánh — đổi ít nhất 1 tuỳ chọn rồi chạy lại.")
+        else:
             st.session_state["g_runs"].append(cfg)
     if b2.button("Xoá hết", use_container_width=True):
```

- [ ] **Step 4: Run** the same command → `3 passed`; `pytest -q` → `98 passed`

- [ ] **Step 5: Commit**

```bash
git add pages/12_Generalization.py tests/test_pages.py
git commit -m "Keep page 12 controls across page switches and toast on duplicate runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Page 13 — Vì sao là CNN?

**Files:** Create `pages/13_Vi_sao_CNN.py`

- [ ] **Step 1: RED** — `pytest tests/test_pages.py -q -k 13_` → `deselected` (the page does not exist yet).

- [ ] **Step 2: Create the page** — `pages/13_Vi_sao_CNN.py`:

```python
import numpy as np
import pandas as pd
import streamlit as st

from cnn_core.layers import MaxPool2D
from cnn_core.shapes import conv_output_shape
from cnn_core.train import (
    MAX_TRAIN_SIZE,
    TrainConfig,
    build_model,
    count_params,
    evaluate,
    load_digits_split,
    shift_images,
)
from viz.matrix_view import fig_feature_maps, fig_lines
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[13]

st.set_page_config(page_title="Vì sao là CNN — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="13. Vì sao là CNN?",
    subtitle="Một lớp Dense cũng nhân-cộng được mọi pixel. Cái làm CNN khác là 3 giả định về ảnh — và chúng đáng giá bao nhiêu.",
    badge="Phần 3 · Bản chất",
    color=COLOR,
)
callout(
    "CNN = MLP bị <b>ràng buộc</b> theo 3 giả định về ảnh: (1) <b>locality</b> — 1 "
    "pixel chủ yếu liên quan tới hàng xóm của nó; (2) <b>weight sharing</b> — 1 "
    "pattern (cạnh, góc) có ý nghĩa như nhau ở mọi vị trí; (3) hệ quả của (2): "
    "<b>equivariance</b> — dịch ảnh thì feature map dịch theo. Ràng buộc đúng = bớt "
    "thứ phải học.",
    color=COLOR,
    label="Ý chính",
)

# ---------------------------------------------------------------------------
# (a) Đếm tham số
# ---------------------------------------------------------------------------
st.markdown("#### (a) Locality + weight sharing — đếm tham số")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    size = c1.select_slider("Ảnh vào (H = W)", options=[8, 28, 64, 224], value=28)
    c_in = c2.select_slider("Kênh vào", options=[1, 3], value=1)
    c_out = c3.select_slider("Số kernel (kênh ra)", options=[4, 8, 16, 32, 64], value=8)
    k = c4.select_slider("Kernel", options=[3, 5], value=3)

    h_out = conv_output_shape(size, k, stride=1, padding=0)
    conv_params = k * k * c_in * c_out + c_out
    n_in, n_out = size * size * c_in, h_out * h_out * c_out
    dense_params = n_in * n_out + n_out

    m1, m2, m3 = st.columns(3)
    m1.metric("Conv — tham số", f"{conv_params:,}")
    m2.metric("Dense cùng shape — tham số", f"{dense_params:,}")
    m3.metric("Dense / Conv", f"{dense_params / conv_params:,.0f}×")
    st.table(
        pd.DataFrame(
            {
                "Mỗi output nhìn bao nhiêu input": [f"{k}×{k}×{c_in} = {k * k * c_in}", f"{size}×{size}×{c_in} = {n_in:,}"],
                "Số bộ trọng số riêng": [f"{c_out} (dùng chung cho {h_out}×{h_out} vị trí)", f"{n_out:,} (mỗi output 1 bộ)"],
                "Bộ nhớ float32": [f"{conv_params * 4 / 1e3:,.1f} KB", f"{dense_params * 4 / 1e9:,.2f} GB"],
            },
            index=["Conv", "Dense"],
        )
    )
    st.caption(
        f"Input {size}×{size}×{c_in} → output {h_out}×{h_out}×{c_out} (H_out = {size} − {k} + 1 = {h_out}, "
        "công thức trang 4). Tham số của conv KHÔNG phụ thuộc kích thước ảnh; của Dense tăng theo "
        "H⁴ — thử 224×224×3."
    )

# ---------------------------------------------------------------------------
# (b) Equivariance
# ---------------------------------------------------------------------------
st.markdown("#### (b) Equivariance — dịch ảnh thì feature map dịch theo")
model, _ = cached_train(TrainConfig())
conv1 = model[0]  # Conv 1 đã train: 16 kernel 3×3, padding 1
_, _, x_val, y_val = load_digits_split()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    idx = c1.slider("Ảnh val", 0, len(x_val) - 1, 0)
    dy = c2.slider("Dịch xuống (px)", -3, 3, 2)
    dx = c3.slider("Dịch sang phải (px)", -3, 3, 1)
    channel = c4.slider("Kernel xem", 0, conv1.params["W"].shape[-1] - 1, 0)

    image = np.kron(x_val[idx, :, :, 0], np.ones((2, 2)))[None, :, :, None]  # phóng 8×8 → 16×16
    moved = shift_images(image, dy, dx)
    f, f_moved = conv1.forward(image, train=False), conv1.forward(moved, train=False)
    diff = shift_images(f, dy, dx) - f_moved
    m = max(abs(dy), abs(dx)) + 1  # bỏ dải viền: ở đó padding / phần bị cắt làm 2 bên khác nhau
    interior_err = float(np.abs(diff[:, m:-m, m:-m]).max())

    st.pyplot(
        fig_feature_maps(
            np.stack([image[0, :, :, 0], moved[0, :, :, 0]], axis=-1),
            titles=["ảnh x", f"ảnh dịch ({dy}, {dx})"],
            max_cols=2,
            cmap="gray_r",
        )
    )
    st.pyplot(
        fig_feature_maps(
            np.stack([f[0, :, :, channel], f_moved[0, :, :, channel], diff[0, :, :, channel]], axis=-1),
            titles=["conv(x)", "conv(dịch x)", "dịch(conv x) − conv(dịch x)"],
            max_cols=3,
            shared_scale=True,
        )
    )
    st.caption(
        f"Sai khác lớn nhất ở vùng trong (bỏ {m} px viền): {interior_err:.1e} — đúng bằng 0 (sai số float). "
        "Chỉ viền khác nhau, vì phần ảnh trượt ra ngoài bị cắt và padding điền 0."
    )

    pool = MaxPool2D(2, 2)
    shifts = [0, 1, 2, 3, 4]
    conv_err, pool_err = [], []
    for s in shifts:
        fs = conv1.forward(shift_images(image, 0, s), train=False)
        crop = slice(s + 1, -(s + 1))
        ref = shift_images(f, 0, s)
        conv_err.append(float(np.linalg.norm((ref - fs)[:, crop, crop]) / np.linalg.norm(f[:, crop, crop])))
        p0, ps = pool.forward(f), pool.forward(fs)
        pool_err.append(float(np.linalg.norm(ps - p0) / np.linalg.norm(p0)))
    st.pyplot(
        fig_lines(
            {
                "conv: ‖dịch(f(x)) − f(dịch x)‖ / ‖f‖ (equivariance)": conv_err,
                "sau max pool: ‖pool(f(dịch x)) − pool(f(x))‖ / ‖pool‖ (invariance)": pool_err,
            },
            xlabels=[f"dịch {s}px" for s in shifts],
            ylabel="sai khác tương đối",
            figsize=(9, 3.2),
        )
    )
    st.caption(
        f"Conv luôn equivariant (đường dưới ≈ 0). Max pool 2×2 chỉ cho bất biến MỘT PHẦN: dịch 1px "
        f"đã làm map sau pool đổi {pool_err[1]:.0%}, dịch 4px đổi {pool_err[4]:.0%}. CNN không tự "
        "động bất biến với dịch chuyển — phần (c) đo hậu quả."
    )

# ---------------------------------------------------------------------------
# (c) CNN vs MLP
# ---------------------------------------------------------------------------
st.markdown("#### (c) Train CNN vs MLP trên cùng dữ liệu")
EPOCHS = {50: 60, 100: 40, 300: 30, MAX_TRAIN_SIZE: 15}  # ít ảnh → nhiều epoch hơn cho đủ số bước
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

with st.container(border=True):
    use_bn = st.toggle("BatchNorm (cả hai mạng)", value=True)
    rows = []
    for n, epochs in EPOCHS.items():
        row = {"Số ảnh train": n}
        for arch in ("cnn", "mlp"):
            net, history = cached_train(TrainConfig(arch=arch, train_size=n, epochs=epochs, use_bn=use_bn))
            shifted = np.mean([evaluate(net, shift_images(x_val, a, b), y_val)[1] for a, b in DIRECTIONS])
            row[f"{arch.upper()} — val"] = history["val_acc"][-1]
            row[f"{arch.upper()} — val dịch 1px"] = float(shifted)
        rows.append(row)
    table = pd.DataFrame(rows)

    cnn_params = count_params(build_model(TrainConfig(use_bn=use_bn)))
    mlp_params = count_params(build_model(TrainConfig(arch="mlp", use_bn=use_bn)))
    m1, m2 = st.columns(2)
    m1.metric("CNN — tham số", f"{cnn_params:,}")
    m2.metric("MLP — tham số", f"{mlp_params:,}", help="Cố ý cho MLP nhiều tham số hơn để so sánh công bằng với nó.")

    sizes = [str(n) for n in EPOCHS]
    c1, c2 = st.columns(2)
    c1.pyplot(
        fig_lines(
            {"CNN": list(table["CNN — val"]), "MLP": list(table["MLP — val"])},
            xlabels=sizes,
            xlabel="số ảnh train",
            ylabel="val acc — ảnh gốc",
        )
    )
    c2.pyplot(
        fig_lines(
            {"CNN": list(table["CNN — val dịch 1px"]), "MLP": list(table["MLP — val dịch 1px"])},
            xlabels=sizes,
            xlabel="số ảnh train",
            ylabel="val acc — ảnh dịch 1px",
        )
    )
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c != "Số ảnh train"}), hide_index=True)

    first, last = table.iloc[0], table.iloc[-1]
    gap_small = first["CNN — val"] - first["MLP — val"]
    gap_full = last["CNN — val"] - last["MLP — val"]
    gap_shift = last["CNN — val dịch 1px"] - last["MLP — val dịch 1px"]
    st.caption(
        f"Ảnh gốc: CNN − MLP = {gap_small:+.1%} với {int(first['Số ảnh train'])} ảnh, {gap_full:+.1%} với "
        f"{int(last['Số ảnh train'])} ảnh. Ảnh dịch 1px: {gap_shift:+.1%} "
        f"(CNN {last['CNN — val dịch 1px']:.0%} vs MLP {last['MLP — val dịch 1px']:.0%}). "
        f"CNN dùng ít hơn {mlp_params - cnn_params:,} tham số."
    )
    callout(
        "Digits của sklearn đã được <b>căn giữa và chuẩn hoá</b> sẵn, nên trên ảnh gốc MLP "
        "gần theo kịp — ở đây locality không cứu được nhiều vì ảnh chỉ có 64 pixel. Lợi thế "
        "thật của CNN hiện ở 2 chỗ: <b>ít tham số hơn</b> cho cùng độ chính xác, và <b>chịu dịch "
        "chuyển tốt hơn hẳn</b> vì mỗi kernel đã thấy pattern ở mọi vị trí. Nhưng CNN cũng rơi "
        "mạnh khi dịch: 1px là 12% bề rộng ảnh 8×8, max pool chỉ bất biến cục bộ, còn lớp Dense "
        "cuối vẫn nhìn vị trí tuyệt đối. Vì vậy vẫn cần augmentation (trang 12); CNN hiện đại "
        "thay Dense cuối bằng global average pooling.",
        color=COLOR,
        label="Đọc kết quả cho đúng",
    )
```

- [ ] **Step 3: Verify, including the BN-off case (Review Focus #5)**

```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/13_Vi_sao_CNN.py", default_timeout=180).run()
assert not at.exception
print([c.value[:120] for c in at.caption if c.value.startswith(("Sai khác", "Ảnh gốc"))])
at.toggle[0].set_value(False).run()
assert not at.exception
print([c.value[:120] for c in at.caption if c.value.startswith("Ảnh gốc")])
EOF
```
Expected: interior error `0.0e+00`. With BN on: `+1.0% với 50 ảnh`, and shifted images about `CNN 70% vs MLP 46%`. With BN off: the gap at 50 images is signed (about `-1.6%`). Then `pytest -q` → `99 passed`.

- [ ] **Step 4: Commit**

```bash
git add pages/13_Vi_sao_CNN.py
git commit -m "Add why-CNN page: parameter count, equivariance, CNN vs MLP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Page 14 — Mạng nhìn thấy gì?

**Files:** Create `pages/14_Mang_nhin_thay_gi.py`

- [ ] **Step 1: RED** — `pytest tests/test_pages.py -q -k 14_` → `deselected`.

- [ ] **Step 2: Create the page** — `pages/14_Mang_nhin_thay_gi.py`:

```python
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cnn_core.analysis import (
    center_concentration,
    confusion_matrix,
    nearest_centroid_accuracy,
    occlusion_map,
    pca_2d,
    receptive_field,
    receptive_field_map,
    saliency_map,
)
from cnn_core.layers import Conv2D, MaxPool2D, ReLU
from cnn_core.train import TrainConfig, build_model, forward_trace, load_digits_split, predict
from viz.matrix_view import fig_feature_maps, fig_lines, fig_matrix, fig_scatter_grid
from viz.theme import PAGE_COLORS, callout, hero, inject_base_css
from viz.widgets import cached_train

COLOR = PAGE_COLORS[14]

st.set_page_config(page_title="Mạng nhìn thấy gì — CNN Visualizer", layout="wide")
inject_base_css()

hero(
    title="14. Mạng nhìn thấy gì?",
    subtitle="Mỗi neuron nhìn vùng nào của ảnh, cả tập dữ liệu biến đổi ra sao qua từng lớp, và pixel nào quyết định câu trả lời.",
    badge="Phần 3 · Bản chất",
    color=COLOR,
)

# ---------------------------------------------------------------------------
# (a) Receptive field
# ---------------------------------------------------------------------------
st.markdown("#### (a) Receptive field — càng sâu, mỗi neuron nhìn vùng càng rộng")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    blocks = c1.slider("Số block Conv → ReLU → [Pool]", 1, 4, 3)
    k = c2.select_slider("Kernel", options=[3, 5], value=3)
    use_pool = c3.toggle("Max pool 2×2 sau mỗi conv", value=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 96, 96, 1))  # trung bình nhiều ảnh random cho gradient mượt
    layers, specs, ch = [], [], 1
    maps, titles = [], []
    for depth in range(1, blocks + 1):
        layers += [Conv2D(ch, 4, k, padding=k // 2, rng=rng), ReLU()]
        specs.append((k, 1))
        if use_pool:
            layers.append(MaxPool2D(2, 2))
            specs.append((2, 2))
        ch = 4
        grad = receptive_field_map(layers, x)
        rows, cols = np.nonzero(grad > 0)
        measured = int(rows.max() - rows.min() + 1)
        maps.append(grad / grad.max())
        titles.append(f"{depth} block · lý thuyết {receptive_field(specs)} · đo {measured} px")

    # cắt mọi map theo vùng của lớp sâu nhất (+ 4 px lề) để thấy vùng nhìn lớn dần
    rows, cols = np.nonzero(maps[-1] > 0)
    window = (
        slice(max(rows.min() - 4, 0), rows.max() + 5),
        slice(max(cols.min() - 4, 0), cols.max() + 5),
    )
    st.pyplot(fig_feature_maps(np.stack([m[window] for m in maps], axis=-1), titles=titles, max_cols=4))

    r = receptive_field(specs)
    share, area, edge = center_concentration(maps[-1], r)
    profile = maps[-1][rows.min() + r // 2, cols.min() : cols.min() + r]
    st.pyplot(
        fig_lines(
            {"|gradient| trên hàng qua tâm": list(profile / profile.max())},
            xlabels=[str(i - r // 2) for i in range(r)] if r <= 40 else None,
            xlabel="vị trí so với tâm (px)",
            ylabel="ảnh hưởng (chuẩn hoá)",
            figsize=(9, 3.0),
        )
    )
    spread = (
        "ảnh hưởng dồn rõ về giữa"
        if share > 1.5 * area
        else "ảnh hưởng còn khá đều — thêm block để thấy nó dồn về giữa"
    )
    st.caption(
        f"Công thức: r ← r + (k − 1)·jump, jump ← jump·stride — pool nhân đôi bước nhảy nên vùng "
        f"nhìn tăng rất nhanh. Nhưng vùng {r}×{r} không đều: hình vuông giữa chiếm {area:.0%} diện "
        f"tích mà mang {share:.0%} ảnh hưởng, mép vùng chỉ còn {edge:.0%} so với tâm — {spread}. "
        "Pixel ở giữa có nhiều đường đi tới neuron hơn pixel ở mép, nên receptive field "
        "<i>hiệu dụng</i> nhỏ hơn lý thuyết và càng sâu càng dồn về giữa (Luo et al., 2016)."
    )

# ---------------------------------------------------------------------------
# (b) Biểu diễn qua từng lớp
# ---------------------------------------------------------------------------
st.markdown("#### (b) Cả tập val đi qua từng lớp — các lớp chữ số tách nhau dần")
trained, _ = cached_train(TrainConfig())
random_net = build_model(TrainConfig())  # cùng kiến trúc, trọng số lúc khởi tạo
x_tr, y_tr, x_val, y_val = load_digits_split(500)

with st.container(border=True):

    def stages(model):
        trace_tr, trace_val = forward_trace(model, x_tr), forward_trace(model, x_val)
        names = [n for n, _ in trace_val]
        picks = [0] + [i for i, n in enumerate(names) if n == "MaxPool2D"] + [len(names) - 2, len(names) - 1]
        return [(trace_tr[i][1], trace_val[i][1]) for i in picks]

    labels = ["Pixel", "Pool 1", "Pool 2", "Dense ẩn", "Logits"]
    rows_data = {"Random": stages(random_net), "Đã train": stages(trained)}
    panels, panel_titles, accs = [], [], {}
    for row_name, feats in rows_data.items():
        accs[row_name] = []
        for label, (f_tr, f_val) in zip(labels, feats):
            acc = nearest_centroid_accuracy(f_tr, y_tr, f_val, y_val)
            accs[row_name].append(acc)
            panels.append(pca_2d(f_val))
            panel_titles.append(f"{row_name} · {label} ({f_val[0].size} chiều)\ntách lớp {acc:.0%}")
    st.pyplot(fig_scatter_grid(panels, y_val, panel_titles, rows=2))
    st.pyplot(fig_lines(accs, xlabels=labels, ylabel="nearest-centroid acc", figsize=(9, 3.0)))
    st.caption(
        f"Mỗi điểm là 1 ảnh val, chiếu PCA xuống 2 chiều, màu = chữ số thật. 'Tách lớp' = gán mỗi "
        f"ảnh cho lớp có tâm gần nhất trong không gian ĐẦY ĐỦ của lớp đó. Mạng đã train: "
        f"{accs['Đã train'][0]:.0%} → {accs['Đã train'][-1]:.0%}. Mạng random: "
        f"{accs['Random'][0]:.0%} → {accs['Random'][-1]:.0%}."
    )
    callout(
        "Mỗi lớp là 1 phép biến đổi không gian. Mạng đã train uốn dữ liệu sao cho ảnh cùng "
        "chữ số co cụm lại, khác chữ số đẩy ra xa — tới logits thì chỉ cần 1 đường thẳng là "
        "tách được. Mạng random cũng biến đổi dữ liệu, nhưng không có mục tiêu nên cấu trúc "
        "ban đầu còn bị xoá dần. PCA chỉ giữ 2 trong nhiều chiều nên cụm trông chồng lên nhau "
        "hơn thực tế — tin con số 'tách lớp' hơn hình.",
        color=COLOR,
        label="Đọc hình",
    )

# ---------------------------------------------------------------------------
# (c) Pixel nào quyết định
# ---------------------------------------------------------------------------
st.markdown("#### (c) Pixel nào quyết định dự đoán")
MODELS = {
    "Mặc định (1297 ảnh sạch)": TrainConfig(),
    "Overfit (300 ảnh, 30% nhãn sai)": TrainConfig(train_size=300, label_noise=0.3, use_bn=False, epochs=60),
}
with st.container(border=True):
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Mô hình", list(MODELS))
    model, _ = cached_train(MODELS[choice])
    probs_all = predict(model, x_val)
    pred_all = probs_all.argmax(axis=1)
    wrong = np.flatnonzero(pred_all != y_val)
    idx = c2.slider("Ảnh val", 0, len(x_val) - 1, int(wrong[0]) if len(wrong) else 0)
    image, label = x_val[idx], int(y_val[idx])
    pred = int(pred_all[idx])

    occ = occlusion_map(model, image, label, patch=2)
    sal = saliency_map(model, image, label)
    p1, p2, p3 = st.columns(3)
    with p1:
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(image[:, :, 0], cmap="gray_r")
        ax.set_title(f"đúng {label} · đoán {pred} ({probs_all[idx, pred]:.0%})", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
    p2.pyplot(fig_feature_maps(occ[:, :, None], titles=[f"Occlusion: che 2×2 → P({label}) rơi"], max_cols=1, shared_scale=True))
    p3.pyplot(fig_feature_maps(sal[:, :, None], titles=[f"Saliency |∂logit {label} / ∂pixel|"], max_cols=1))
    st.caption(
        "Occlusion: vùng đậm = thiếu nó thì mạng mất tự tin vào lớp đúng (số âm = che đi lại "
        "tự tin hơn). Saliency: pixel nào chỉ cần nhích 1 chút là logit đổi nhiều. Hai cách hỏi "
        "khác nhau nên không nhất thiết trùng. Mặc định chọn sẵn 1 ảnh bị đoán sai (nếu có)."
    )

    cm = confusion_matrix(y_val, pred_all)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:3]
    c_cm, c_txt = st.columns([1.3, 1])
    c_cm.pyplot(fig_matrix(cm, title="Nhãn đúng (hàng) × dự đoán (cột)", figsize=(5.2, 5.2)))
    with c_txt:
        st.metric("Val acc", f"{np.mean(pred_all == y_val):.1%}", help=f"{len(wrong)} / {len(y_val)} ảnh sai")
        if top:
            st.markdown("**Nhầm nhiều nhất**\n" + "\n".join(f"- {n} ảnh **{i}** bị đoán là **{j}**" for n, i, j in top))
        else:
            st.markdown("Không nhầm ảnh nào.")
        st.caption("Đổi sang mô hình overfit ở trên để so: nó nhầm nhiều hơn hẳn — và nhầm ở những cặp nào.")
```

- [ ] **Step 3: Verify at extreme settings (Review Focus #4)**

```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/14_Mang_nhin_thay_gi.py", default_timeout=180).run()
at.select_slider[0].set_value(5).run(); at.slider[0].set_value(4).run()
assert not at.exception
print([c.value[:110] for c in at.caption if c.value.startswith("Công thức")])
at.selectbox[0].set_value("Overfit (300 ảnh, 30% nhãn sai)").run()
assert not at.exception
print([m.value for m in at.metric], [m.value for m in at.markdown if m.value.startswith("**Nhầm")])
EOF
```
Expected: the caption shows the 76×76 region; the overfit model reaches about 75.6% val acc and lists 3 frequently confused pairs. Then `pytest -q` → `100 passed`.

- [ ] **Step 4: Commit**

```bash
git add pages/14_Mang_nhin_thay_gi.py
git commit -m "Add what-the-network-sees page: receptive field, representations, saliency

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Home page + docs

**Files:** Modify `app.py`, `README.md`, `TODO.md`

- [ ] **Step 1: Apply** — Save the block to a file and run `git apply` on it.

```diff
diff --git a/app.py b/app.py
index ced8496..b35f966 100644
--- a/app.py
+++ b/app.py
@@ -26,5 +26,7 @@ st.markdown(
     <b>Phần 2 (trang 9–12)</b>: <code>cnn_core/layers.py</code> + <code>train.py</code> (numpy vectorized,
     viết sẵn, có backward) train 1 CNN thật trên ảnh chữ số 8×8 — để thấy BatchNorm, loss,
-    backprop và overfit bằng số thật.
+    backprop và overfit bằng số thật.<br>
+    <b>Phần 3 (trang 13–14)</b>: bản chất CNN — vì sao conv thắng Dense, mỗi neuron nhìn
+    vùng nào, và dữ liệu biến đổi ra sao qua từng lớp (<code>cnn_core/analysis.py</code>).
     </div>
     """,
@@ -100,4 +102,12 @@ for num in (10, 11, 12):  # trang 9–12 dùng chung layers.py + train.py
     ready[num] = ready[9]
 
+try:
+    from cnn_core.analysis import receptive_field
+
+    ready[13] = ready[9] and _check(build_model, TrainConfig(arch="mlp"))
+    ready[14] = ready[9] and _check(receptive_field, [(3, 1), (2, 2)])
+except ImportError:
+    ready[13] = ready[14] = False
+
 PAGES = [
     (1, "pages/1_Kernel_va_Convolution.py", "Kernel & Convolution", "Trượt kernel qua input, nhân-cộng từng vị trí."),
@@ -113,7 +123,9 @@ PAGES = [
     (11, "pages/11_Backprop_Training.py", "Backprop & Training", "Gradient chảy ngược, đo xem mỗi lớp thật sự học được bao nhiêu."),
     (12, "pages/12_Generalization.py", "Generalization", "Vì sao dropout, weight decay, augmentation kéo val lên."),
+    (13, "pages/13_Vi_sao_CNN.py", "Vì sao là CNN?", "Locality, weight sharing, equivariance — so thẳng với MLP."),
+    (14, "pages/14_Mang_nhin_thay_gi.py", "Mạng nhìn thấy gì?", "Receptive field, dữ liệu qua từng lớp, pixel nào quyết định."),
 ]
 
-st.markdown("### Hành trình 12 bước")
+st.markdown("### Hành trình 14 bước")
 
 cols = st.columns(4)
```

```diff
diff --git a/README.md b/README.md
index 4f0c8e9..6a574d0 100644
--- a/README.md
+++ b/README.md
@@ -28,5 +28,5 @@ thuộc.
 ## Tính năng
 
-App gồm 12 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học. Trang 1–8 (phần 1) là forward pass với kernel random; trang 9–12 (phần 2) train thật 1 CNN nhỏ để thấy BatchNorm, loss, backprop và overfit:
+App gồm 14 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học. Trang 1–8 (phần 1) là forward pass với kernel random; trang 9–12 (phần 2) train thật 1 CNN nhỏ để thấy BatchNorm, loss, backprop và overfit; trang 13–14 (phần 3) đi vào bản chất: vì sao conv thắng Dense và mạng thật sự nhìn thấy gì:
 
 | # | Trang | Nội dung |
@@ -40,8 +40,10 @@ App gồm 12 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo th
 | 7 | Conv Block | Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh (RGB), quan sát nhiều feature map ra từ nhiều kernel, histogram giá trị sau từng bước |
 | 8 | Full Pipeline | Xếp 2-3 Conv Block liên tiếp, forward 1 ảnh thật qua từng bước Conv / ReLU / Pool của từng block, theo dõi shape thu nhỏ dần và std activation co lại qua độ sâu |
-| 9 | BatchNorm | Tính μ, σ², x̂, γx̂+β bằng số trên 1 mini-batch; stack 10 lớp conv random có/không BN; train mode vs eval mode (running stats); BN giúp train nhanh hơn |
+| 9 | BatchNorm | Tính μ, σ², x̂, γx̂+β bằng số trên 1 mini-batch; stack tới 10 lớp conv random có/không BN; train mode vs eval mode (running stats); BN giúp train nhanh hơn |
 | 10 | Softmax + Cross-Entropy | Kéo logits xem xác suất, loss `-log p` và gradient `p − y`; 1 ảnh chữ số đi hết mạng đã train tới tận loss |
 | 11 | Backprop & Training | Đường học, ảnh hưởng của learning rate, gradient norm từng lớp, kernel và feature map trước vs sau khi train |
 | 12 | Generalization | Overfit có chủ đích (ít data + nhãn sai), so sánh chồng dropout / weight decay / augmentation / BatchNorm / thêm data, early stopping, ảnh val bị đoán sai |
+| 13 | Vì sao là CNN? | Đếm tham số conv vs Dense cùng shape; equivariance (dịch ảnh → feature map dịch theo, max pool chỉ bất biến một phần); train CNN vs MLP nhiều tham số hơn trên cùng data, đo trên ảnh gốc và ảnh dịch 1px |
+| 14 | Mạng nhìn thấy gì? | Receptive field lý thuyết vs đo bằng gradient, effective RF dồn về giữa; PCA của cả tập val qua từng lớp (mạng đã train vs random); occlusion, saliency, confusion matrix |
 
 ## Kiến trúc & nguyên tắc thiết kế
@@ -118,5 +120,6 @@ CNN_Visualizer/
 │   ├── block.py                   # ConvBlock — Conv → ReLU → Pool, make_random_kernels
 │   ├── layers.py                  # Conv2D, BatchNorm2D, Dropout, Dense... có forward + backward (vectorized)
-│   └── train.py                   # dataset digits, build_model, SGD momentum + weight decay, train()
+│   ├── train.py                   # dataset digits, build_model (cnn | mlp), SGD momentum + weight decay, train()
+│   └── analysis.py                # receptive field, PCA, nearest-centroid, occlusion, saliency, confusion matrix
 ├── viz/
 │   ├── theme.py                   # bảng màu, hero header, card, step-progress dots
@@ -195,5 +198,6 @@ matmul/max; backward cộng dồn gradient ngược về từng pixel gốc
 running stats khi eval. `softmax_cross_entropy_backward` trả về `(p − y)/N`.
 
-Mọi backward được kiểm bằng gradient check (sai phân trung tâm), `Conv2D`
+Mọi backward (trừ `Dropout`, kiểm bằng test hành vi) được kiểm bằng gradient
+check (sai phân trung tâm) — `BatchNorm2D` ở cả train mode lẫn eval mode. `Conv2D`
 được đối chiếu với `conv2d_multichannel` vòng lặp tay, và (nếu có `torch`)
 với `torch.nn.functional.conv2d` / `batch_norm`.
@@ -206,5 +210,15 @@ Dataset `load_digits` (8×8, 10 lớp), 500 ảnh val cố định. Mô hình:
 SGD momentum 0.9, weight decay chỉ áp cho `W`. Tuỳ chọn `label_noise` đổi 1 tỉ lệ
 ảnh train sang nhãn sai ngẫu nhiên để mô phỏng dữ liệu bẩn. Train full data
-~1 giây, val acc ~99%.
+~1 giây, val acc ~99%. `arch="mlp"` dựng MLP 24k tham số (nhiều hơn CNN) để
+so sánh ở trang 13.
+
+### Phân tích mô hình (`cnn_core/analysis.py`)
+
+`receptive_field` tính vùng nhìn lý thuyết (`r ← r + (k − 1)·jump`,
+`jump ← jump·stride`); `receptive_field_map` đo nó bằng gradient của 1 neuron
+giữa về input. `pca_2d` + `nearest_centroid_accuracy` đo các lớp chữ số tách
+nhau thế nào ở từng lớp. `occlusion_map` (che từng ô, xem xác suất rơi) và
+`saliency_map` (|∂logit/∂pixel|, chạy trên bản sao model để không đụng model
+đang cache) chỉ ra pixel nào quyết định dự đoán.
 
 ## Kiểm thử
@@ -222,7 +236,7 @@ pytest -v
 ## Trạng thái hoàn thành
 
-Toàn bộ 6 phase trong `TODO.md` đã hoàn tất, cả 12 trang Streamlit chạy
+Toàn bộ 6 phase trong `TODO.md` đã hoàn tất, cả 14 trang Streamlit chạy
 được — từ input tuỳ chỉnh, qua ảnh thật đi qua nhiều Conv Block, tới train
-thật và so sánh các kỹ thuật chống overfit.
+thật, so sánh các kỹ thuật chống overfit, và soi xem mạng nhìn thấy gì.
 
 ## Định hướng mở rộng
```

```diff
diff --git a/TODO.md b/TODO.md
index 0cbb88f..4099a3c 100644
--- a/TODO.md
+++ b/TODO.md
@@ -109,4 +109,7 @@ CNN — pooling/block/pipeline chỉ là lắp ráp lại đúng 2 khối này.
 - [x] Phần 2: BatchNorm, Softmax + Cross-Entropy, Backprop & Training,
       Generalization (trang 9–12, `cnn_core/layers.py` + `train.py`)
+- [x] Phần 3: Vì sao là CNN (so với MLP), Mạng nhìn thấy gì (receptive
+      field, biểu diễn qua từng lớp, occlusion/saliency) — trang 13–14,
+      `cnn_core/analysis.py`
 - [ ] Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur ở trang 1 để xây
       trực giác "kernel = bộ dò 1 loại pattern"
```

- [ ] **Step 2: Verify** — `pytest -q` → `100 passed`. The home page shows `Tiến độ tổng: 14/14`:

```bash
python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120).run()
print([m.value for m in at.markdown if "Tiến độ" in m.value], at.exception)
EOF
```

- [ ] **Step 3: Commit**

```bash
git add app.py README.md TODO.md
git commit -m "List pages 13-14 on home page and document analysis module

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
