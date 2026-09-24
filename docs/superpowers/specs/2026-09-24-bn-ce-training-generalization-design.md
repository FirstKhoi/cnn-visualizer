# Design — BatchNorm, Softmax/CE, Training, Generalization + sửa trang cũ

Ngày: 2026-09-24

## Mục tiêu

App hiện tại chỉ có forward pass với kernel random (trang 1–8). Đợt này thêm
phần "bản chất" còn thiếu: vì sao cần BatchNorm, dữ liệu đi tới loss như thế
nào, mạng học ra sao (backprop), và vì sao các kỹ thuật regularization kéo
đường validation lên. Đồng thời sửa trang cũ để trực quan trung thực hơn.

Thành công khi:
- 4 trang mới (9–12) chạy được, mỗi lần train < ~5 giây trên laptop.
- Trang 8 cho thấy rõ activation co lại qua từng block (hiện đang bị che vì
  mỗi feature map tự scale màu riêng).
- `pytest` pass, bao gồm gradient check cho mọi lớp có backward.

## Quyết định đã chốt

- Toàn bộ toán mới do Claude viết bằng **numpy** (không torch trong app).
  Torch chỉ dùng trong test để đối chiếu, qua `pytest.importorskip`.
- Code mới **vectorized** (im2col) vì phải train; code cũ giữ nguyên vòng
  lặp tay để dạy cơ chế. Test đối chiếu `Conv2D` mới với `conv2d` cũ.
- Dataset: `sklearn.datasets.load_digits` (1797 ảnh 8×8 xám, 10 lớp),
  không cần tải mạng.
- Biểu đồ vẫn dùng matplotlib + palette trong `viz/theme.py`.

## Kiến trúc

### `cnn_core/layers.py` (mới)

Mọi lớp nhận batch NHWC (hoặc `(N, D)` với Dense), có `forward(x, train)` và
`backward(dout)`; lớp có tham số lưu `params` và `grads` dạng dict cùng key.

- `Conv2D(in_ch, out_ch, k, stride=1, padding=0)` — im2col + matmul, He init.
- `ReLU`, `MaxPool2D(size, stride)` (lưu mask argmax cho backward),
  `Flatten`, `Dense(in, out)`, `Dropout(p)` (inverted dropout, chỉ khi train).
- `BatchNorm2D(ch, momentum=0.9, eps=1e-5)` — train: dùng μ/σ² của batch,
  cập nhật running stats; eval: dùng running stats. Thống kê trên trục (N,H,W).
- `softmax(logits)` ổn định số (trừ max), `cross_entropy(probs, y)` trả về
  loss trung bình; `softmax_cross_entropy_backward(probs, y) = (p − onehot)/N`.

### `cnn_core/train.py` (mới)

- `load_digits_split(train_size, seed)` → `(x_train, y_train, x_val, y_val)`,
  ảnh shape `(N, 8, 8, 1)` chuẩn hoá về [0, 1]; val cố định 500 ảnh.
- `TrainConfig` (dataclass): `use_bn`, `dropout`, `weight_decay`, `augment`,
  `train_size`, `epochs`, `lr`, `batch_size`, `seed`.
- `build_model(cfg)` → list lớp:
  `Conv3×3(8,pad1) → [BN] → ReLU → Pool2 → Conv3×3(16,pad1) → [BN] → ReLU →
  Pool2 → Flatten → [Dropout] → Dense(64→10)`.
- `augment_shift(x, rng)` — dịch ngẫu nhiên ±1 px mỗi ảnh (pad 0).
- SGD momentum 0.9 + weight decay (L2 cộng vào grad, không áp cho BN/bias).
- `train(cfg)` → `(model, history)`; history theo epoch gồm
  `train_loss, train_acc, val_loss, val_acc, grad_norms` (theo lớp).
- `predict(model, x)`, `forward_trace(model, x)` (list `(tên lớp, output)`
  cho trang 10/11).

### `cnn_core/block.py`

Thêm `ConvBlock.forward_steps(x)` → `[("conv", y1), ("relu", y2), ("pool", y3)]`;
`forward` gọi lại `forward_steps` và trả về phần tử cuối.

### `viz/matrix_view.py`

- `fig_feature_maps(..., shared_scale=False)` — `True` thì mọi map dùng chung
  vmin/vmax + 1 colorbar.
- `fig_activation_hist(named_arrays)` — histogram chồng/lưới theo lớp, ghi
  std và % zero.
- `fig_curves(runs, metric)` — vẽ train (nét đứt) vs val (nét liền) cho nhiều
  lần chạy, mỗi lần 1 màu.

## Trang

### Sửa trang cũ
- **Trang 7**: bật `shared_scale`, thêm histogram activation.
- **Trang 8**: mỗi block tách 3 bước con Conv/ReLU/Pool (step_controls đi qua
  từng bước con), feature map chung scale, biểu đồ std + % zero qua các lớp,
  callout dẫn sang trang 9.
- **Trang chủ**: 12 thẻ, kiểm tra trạng thái cho `layers` / `train`.

### 9. BatchNorm
- (a) Mini-batch nhỏ (vd 4 mẫu × vài giá trị): hiện μ, σ², x̂, γx̂+β bằng số;
  slider γ, β.
- (b) Stack 6 `Conv2D` random + ReLU trên batch digits, có/không BN: histogram
  từng lớp + đường std theo độ sâu.
- (c) Train vs eval: slider batch size → μ của batch dao động quanh running
  mean (vẽ nhiều batch), giải thích vì sao eval dùng running stats.

### 10. Softmax + Cross-Entropy
- (a) Slider logits (10 lớp hoặc ít hơn), chọn lớp đúng → bar softmax, loss
  `-log p_true`, đường cong `-log p` có điểm hiện tại, bar gradient `p − y`.
- (b) Chọn 1 ảnh digit val, đi qua mạng đã train (cache) theo `forward_trace`:
  ảnh → feature map conv1 → pool → conv2 → vector flatten → logits → probs →
  loss.

### 11. Backprop & Training
- Train config mặc định (cache). Đường loss/acc theo epoch.
- Kernel conv1 trước vs sau train; feature map của cùng 1 ảnh trước vs sau.
- Gradient norm từng lớp theo epoch.

### 12. Generalization
- Controls: train size, augmentation, dropout, weight decay, BN, epochs.
- Nút "Chạy" → thêm run vào `st.session_state`; vẽ chồng mọi run (train nét
  đứt, val nét liền), bảng tóm tắt gap cuối = train acc − val acc; nút xoá.
- Preset gợi ý: "Overfit (100 ảnh, không gì)" rồi lần lượt +aug, +dropout,
  +weight decay, +BN, +data.
- Callout giải thích từng kỹ thuật: aug = thêm dữ liệu hiệu dụng; dropout =
  nhiễu/ensemble; weight decay = trọng số nhỏ → hàm mượt; BN = tối ưu dễ hơn +
  nhiễu nhẹ; thêm data = cách chắc chắn nhất.
- Lưới ảnh val bị đoán sai của run mới nhất.

Train cache bằng `st.cache_data` theo config (trả về history + trọng số dạng
mảng để hash được).

## Test

- `tests/test_layers.py`: gradient check finite-difference cho `Conv2D`,
  `MaxPool2D`, `BatchNorm2D`, `Dense`, `ReLU`, softmax-CE; `Conv2D.forward`
  khớp `conv2d_multichannel` cũ; softmax tổng = 1; BN train output có
  mean≈0/std≈1 mỗi kênh; Dropout eval = identity; torch đối chiếu conv/BN nếu
  có.
- `tests/test_train.py`: train cấu hình mặc định vài epoch → val acc > 0.9;
  `forward_trace` trả đủ lớp với shape đúng.
- `tests/test_block.py`: `forward_steps` khớp `forward`.

## Ngoài phạm vi

Dataset lớn (MNIST/CIFAR), optimizer khác (Adam), learning-rate schedule,
giao diện plotly/tương tác hover.
