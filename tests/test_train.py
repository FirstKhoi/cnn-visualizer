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
