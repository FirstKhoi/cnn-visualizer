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
