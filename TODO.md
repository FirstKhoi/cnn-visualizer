# TODO — CNN Visualizer

Mốc hiện tại: mới scaffold, `cnn_core/` toàn `NotImplementedError`. Đích:
implement từng hàm theo đúng thứ tự phase — mỗi phase mở khóa đúng 1 trang
Streamlit tương ứng trong `pages/`.

Đừng nhảy cóc: `convolution.py` dùng `padding.py`, `block.py` dùng cả ba
(`convolution`, `activation`, `pooling`). Làm sai thứ tự sẽ phải quay lại.

---

## Phase 0 — Khởi động (~15 phút)

- [x] `pip install -r requirements.txt`
- [x] `streamlit run app.py` chạy được, thấy trang chủ (chưa cần implement gì)
- [x] `pytest -v` chạy được — tất cả test đang **fail** vì `NotImplementedError`,
      đó là trạng thái đúng lúc bắt đầu

---

## Phase 1 — Padding (`cnn_core/padding.py`)

- [x] `pad_matrix(matrix, pad, mode="zero")` — thêm viền 0 quanh ma trận 2D
  - *Xong khi:* `pytest tests/test_padding.py` pass
  - *Xong khi:* trang `pages/2_Padding.py` chạy, kéo slider `pad` thấy ma
    trận input to ra đúng `(H+2·pad, W+2·pad)`

---

## Phase 2 — Convolution cơ bản (`cnn_core/convolution.py`)

- [x] `conv2d(input, kernel, stride=1, padding=0)` — 1 kênh, dùng
      `pad_matrix` ở trên, vòng lặp tay (không dùng `np.convolve`/`scipy`)
  - Lưu ý: đây là **cross-correlation** (kernel không bị lật 180°) — đúng
    quy ước CNN, khác định nghĩa "convolution" toán thuần túy
  - *Xong khi:* `pytest tests/test_convolution.py::test_conv2d_basic` pass
  - *Xong khi:* `pages/1_Kernel_va_Convolution.py` chạy — đổi giá trị kernel
    3x3 bằng tay, thấy output đổi theo, có preset "edge detection"/"blur"
    để thấy kernel khác nhau "nhìn" ra gì

- [x] `pytest tests/test_convolution.py::test_conv2d_stride` pass (stride>1)
  - *Xong khi:* `pages/3_Stride.py` chạy — tăng stride thấy output nhỏ dần,
    và thấy rõ những vị trí bị "bỏ qua" trên input

---

## Phase 3 — Công thức shape (`cnn_core/shapes.py`)

- [x] `conv_output_shape(input_size, kernel_size, stride, padding)` →
      `(W - K + 2P) // S + 1`, raise lỗi rõ ràng nếu chia không hết
  - *Xong khi:* `pytest tests/test_shapes.py` pass
  - *Xong khi:* `pages/4_Cong_thuc_Output_Shape.py` chạy — nhập W/K/S/P bằng
    tay, so số app tính với số bạn tính nhẩm; thử 1 bộ số làm shape "vỡ"
    (không chia hết) để thấy lỗi

---

## Phase 4 — Pooling & Activation

- [x] `cnn_core/pooling.py`: `max_pool2d(input, size, stride)`,
      `avg_pool2d(input, size, stride)`
  - *Xong khi:* `pytest tests/test_pooling.py` pass
  - *Xong khi:* `pages/5_Pooling.py` chạy — so sánh trực quan max vs avg
    trên cùng 1 ma trận, thấy max giữ chi tiết sắc nét hơn

- [x] `cnn_core/activation.py`: `relu(x)`, `sigmoid(x)`
  - *Xong khi:* `pytest tests/test_activation.py` pass
  - *Xong khi:* `pages/6_Activation.py` chạy — thấy ReLU cắt hết giá trị âm
    trên feature map (giải thích vì sao cần non-linearity sau conv)

---

## Phase 5 — Ghép Block (`cnn_core/block.py`)

- [x] `conv2d_multichannel(input_hwc, kernels, stride=1, padding=0)` trong
      `convolution.py` — mở rộng `conv2d` sang nhiều kênh input + nhiều
      kernel (mỗi kernel ra 1 feature map)
- [x] `class ConvBlock`: `__init__(kernels, bias, pool_size, pool_stride)`,
      `forward(x)` = conv → relu → pool, dùng lại 3 hàm đã viết ở trên
  - *Xong khi:* `pytest tests/test_block.py` pass
  - *Xong khi:* `pages/7_Conv_Block.py` chạy trên ảnh nhỏ nhiều kênh (RGB),
    thấy N feature map ra từ N kernel ngẫu nhiên

---

## Phase 6 — Full pipeline trên ảnh thật

- [x] `pages/8_Full_Pipeline.py`: xếp 2-3 `ConvBlock` liên tiếp, upload/chọn
      1 ảnh mẫu, forward qua từng block
  - *Xong khi:* thấy bảng shape thu nhỏ dần qua từng block (vd:
    `64x64x3 → 32x32x8 → 16x16x16`), khớp với `conv_output_shape` bạn viết
    ở Phase 3
  - *Xong khi:* xem được feature map ở **mỗi** block, không chỉ block cuối

---

## Nếu chỉ làm được một thứ

**Phase 1 + Phase 2** (padding + convolution cơ bản). Đó là 90% trực giác về
CNN — pooling/block/pipeline chỉ là lắp ráp lại đúng 2 khối này.

---

## Ý mở rộng (sau khi xong hết, tùy hứng)

- [x] So sánh output `conv2d` tự viết với `torch.nn.functional.conv2d` cùng
      kernel — số phải khớp tuyệt đối (sai số float cho phép)
      (`tests/test_layers.py::test_conv_and_batchnorm_match_torch`)
- [x] Phần 2: BatchNorm, Softmax + Cross-Entropy, Backprop & Training,
      Generalization (trang 9–12, `cnn_core/layers.py` + `train.py`)
- [ ] Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur ở trang 1 để xây
      trực giác "kernel = bộ dò 1 loại pattern"
- [ ] Padding `mode="reflect"` bên cạnh zero-padding, so sánh viền ảnh
- [ ] Benchmark tốc độ: vòng lặp tay vs `np.einsum`/vectorize — thấy vì sao
      framework thật không dùng for-loop
