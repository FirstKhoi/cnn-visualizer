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


def test_flatten_gradients():
    from cnn_core.layers import Flatten

    rng = np.random.default_rng(9)
    check_layer_grads(Flatten(), rng.normal(size=(2, 3, 3, 2)), rng)


@pytest.mark.parametrize("shape", [(4, 3, 3, 3), (6, 3)])
def test_batchnorm_eval_mode_gradient(shape):
    """Eval dùng running stats (hằng số) -> gradient khác hẳn công thức train."""
    rng = np.random.default_rng(10)
    bn = BatchNorm2D(3)
    bn.params["gamma"] = rng.normal(size=3)
    bn.running_mean = rng.normal(size=3)
    bn.running_var = rng.uniform(0.5, 2.0, size=3)
    x = rng.normal(size=shape)
    g = rng.normal(size=shape)
    bn.forward(x, train=False)
    dx = bn.backward(g)

    def loss():
        return float(np.sum(bn.forward(x, train=False) * g))

    np.testing.assert_allclose(dx, numeric_grad(loss, x), atol=1e-6)


def test_relu_forward_does_not_read_mask_back_from_attribute():
    """2 phiên dùng chung 1 model đã cache: phiên khác có thể ghi đè self.mask
    giữa lúc gán và lúc dùng. forward phải tính từ biến cục bộ."""

    class Racing(ReLU):
        def __getattribute__(self, name):
            if name == "mask":
                return np.zeros((1,), dtype=bool)  # "mask của phiên khác"
            return super().__getattribute__(name)

    x = np.array([[-1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(Racing().forward(x), [[0.0, 2.0, 3.0]])
