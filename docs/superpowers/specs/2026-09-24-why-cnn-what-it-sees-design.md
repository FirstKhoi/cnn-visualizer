# Design — Trang 13 (Vì sao là CNN?), 14 (Mạng nhìn thấy gì?) + dọn nợ review

Ngày: 2026-09-24. Nối tiếp `2026-09-24-bn-ce-training-generalization-design.md`.

## Mục tiêu

Phần 2 (trang 9–12) đã dạy mạng *học như thế nào*. Còn thiếu hai câu hỏi cốt
lõi về bản chất CNN:

1. **Vì sao là CNN mà không phải MLP?** — inductive bias: locality, weight
   sharing, translation equivariance.
2. **Mạng thật sự "nhìn" gì?** — receptive field, biểu diễn của cả tập dữ liệu
   biến đổi qua từng lớp, pixel nào quyết định dự đoán.

Đồng thời dọn 7 minor còn nợ từ final review đợt trước.

Thành công khi:
- Trang 13, 14 chạy; lần đầu mở mỗi trang < ~10 giây (train có cache), các lần
  sau tức thì.
- Mọi câu giải thích khớp với số liệu chính trang đó đo được (bài học từ review
  trước: copy sai = bug). Chỗ nào số phụ thuộc lựa chọn của người dùng thì câu
  chữ đổi theo số đo.
- `pytest` pass, có test cho mọi hàm phân tích mới và cho từng minor được sửa.

## Quyết định đã chốt

- Vẫn numpy thuần, không thêm dependency.
- MLP so sánh: `Flatten → Dense(64→160) → [BN] → ReLU → Dense(160→80) → [BN] →
  ReLU → [Dropout] → Dense(80→10)` — 24 090 tham số, NHIỀU hơn CNN (13 706) để
  so sánh công bằng theo hướng có lợi cho MLP.
- Công cụ phân tích mô hình gom vào 1 module mới `cnn_core/analysis.py`.
- Phân tích dùng gradient (saliency, receptive field) chạy trên **bản sao**
  model (`copy.deepcopy`) — không bao giờ gọi backward trên model dùng chung
  trong `st.cache_resource`.

## Số đo prototype (quyết định nội dung trang 13)

Trung bình 3 seed, val 500 ảnh; "dịch 1px" = trung bình 4 hướng, viền điền 0.
Epoch theo số ảnh train: 50→60, 100→40, 300→30, 1297→15.

| train | CNN val / dịch 1px | MLP val / dịch 1px |
|---|---|---|
| 50 (BN) | 0.841 / 0.480 | 0.827 / 0.362 |
| 300 (BN) | 0.958 / 0.612 | 0.954 / 0.418 |
| 1297 (BN) | 0.991 / 0.693 | 0.989 / 0.456 |
| 1297 (không BN) | 0.983 / 0.641 | 0.960 / 0.444 |

Kết luận trang 13 phải nói đúng như vậy: trên digits 8×8 **đã căn giữa sẵn**,
MLP gần hoà CNN về accuracy — lợi thế thật của CNN ở đây là **ít tham số hơn**
và **chịu dịch chuyển tốt hơn nhiều**; và CNN cũng rơi mạnh khi dịch (conv chỉ
equivariant; lớp Dense cuối + ảnh nhỏ làm mất bất biến) → augmentation (trang
12) vẫn cần. Với ảnh lớn, số tham số MLP bùng nổ (phần a) nên khoảng cách lớn
hơn nhiều.

Receptive field đo bằng gradient (Conv3×3 pad1 → ReLU → MaxPool2): 4 → 10 → 22
px, khớp công thức. Biểu diễn qua từng lớp (nearest-centroid acc trên val):
mạng đã train 0.886 (pixel) → 0.926 → 0.982 → 0.988; mạng random 0.886 → 0.876
→ 0.858 → 0.636 (logits).

## Kiến trúc

### `cnn_core/layers.py`
- `ReLU.forward`: tính mask vào biến cục bộ rồi mới gán `self.mask` — không
  đọc lại thuộc tính (race khi 2 tab dùng chung model).
- `BatchNorm2D.backward` ở **eval mode**: running stats là hằng số nên
  `dx = dout · γ / √(running_var + ε)` (hiện tại dùng nhầm công thức train).
  Cache ghi lại mode của lần forward gần nhất.

### `cnn_core/train.py`
- `TrainConfig.arch: str = "cnn"` (`"cnn"` | `"mlp"`); `build_model` raise
  `ValueError` với arch lạ.
- `count_params(model) -> int`.
- `shift_images(x, dy, dx) -> ndarray` — dịch cả batch đúng (dy, dx) pixel,
  viền điền 0.

### `cnn_core/analysis.py` (mới)
- `receptive_field(specs: list[tuple[int, int]]) -> int` — specs là
  `(kernel, stride)` từng lớp theo thứ tự; công thức `r += (k−1)·jump;
  jump *= stride`, bắt đầu `r = 1, jump = 1`.
- `receptive_field_map(layers, x) -> ndarray (H, W)` — trung bình |∂ neuron
  giữa / ∂ input| qua batch và kênh (backward từ 1 vị trí trung tâm, mọi kênh).
- `pca_2d(features) -> ndarray (N, 2)` — SVD trên dữ liệu đã trừ mean.
- `nearest_centroid_accuracy(f_train, y_train, f_val, y_val) -> float`.
- `occlusion_map(model, image, label, patch=2) -> ndarray` — với mỗi vị trí
  che ô patch×patch bằng 0, lượng xác suất lớp đúng bị rơi.
- `saliency_map(model, image, label) -> ndarray (H, W)` — |∂ logit lớp đúng /
  ∂ pixel|, chạy trên `deepcopy(model)`.
- `confusion_matrix(y_true, y_pred, n_classes=10) -> ndarray`.

### `viz/`
- `fig_feature_maps(..., cmap=None)`: `None` → tự chọn `DIV` nếu dữ liệu có số
  âm, ngược lại `WARM` (hết cảnh 0 rơi vào giữa colormap tuần tự).
- `fig_scatter_grid(panels, labels, titles, ...)` — lưới scatter 2D tô màu theo
  nhãn (10 màu), dùng cho PCA.

## Trang

### 13. Vì sao là CNN? (`pages/13_Vi_sao_CNN.py`)
- (a) **Đếm tham số**: slider kích thước ảnh, số kênh in/out, kernel. Bảng + bar
  log-scale: conv (`K·K·C_in·C_out + C_out`) vs dense cho cùng shape input →
  output (`(H·W·C_in)·(H_out·W_out·C_out) + H_out·W_out·C_out`, H_out tính bằng
  `conv_output_shape` của Phase 3). Giải thích locality + weight sharing.
- (b) **Equivariance**: 1 ảnh val phóng lên 16×16, slider dịch (dy, dx). Hiện
  ảnh gốc/ảnh dịch, feature map conv 1 (kernel đã train) của cả hai, và hiệu
  `shift(f(x)) − f(shift(x))` = 0 ở vùng trong. Biểu đồ: sai lệch sau conv
  (luôn ≈ 0) vs sau max pool theo độ dịch 0..4 — pool dịch 2px thì map pool
  dịch đúng 1 ô, dịch 1px thì không.
- (c) **Train CNN vs MLP**: cùng data, cùng BN/lr; 4 cỡ train (bảng epoch ở
  trên). Biểu đồ val acc theo số ảnh train (2 đường) và val trên ảnh dịch 1px
  (2 đường), bảng số tham số. Caption dựng từ số đo: nêu chênh lệch thật.

### 14. Mạng nhìn thấy gì? (`pages/14_Mang_nhin_thay_gi.py`)
- (a) **Receptive field**: slider số block (1–4), kernel (3/5), bật/tắt pool.
  Stack Conv→ReLU→[Pool] random trên input 96×96 (16 mẫu). Heatmap
  `receptive_field_map` mỗi độ sâu + "lý thuyết r px · đo được r px"; profile
  1D qua tâm (effective RF dạng Gauss, rìa đóng góp ít).
- (b) **Biểu diễn qua từng lớp**: PCA 2D của 500 ảnh val tại Input, Pool 1,
  Pool 2, Dense ẩn, logits — hàng trên mạng random, hàng dưới mạng đã train, tô
  màu theo nhãn. Đường nearest-centroid acc theo lớp cho cả hai.
- (c) **Pixel nào quyết định**: chọn ảnh val → ảnh, occlusion map, saliency,
  dự đoán; confusion matrix val (10×10) + các cặp nhầm nhiều nhất.

## Dọn nợ review

| Minor | Sửa |
|---|---|
| ReLU race | như trên |
| Colormap tuần tự với số âm (trang 8 conv, trang 10 BN) | `cmap=None` tự chọn |
| Trang 12 "Chạy" config đã có không phản hồi | `st.toast` |
| Trang 12 control reset khi rời trang | lưu bản sao không-phải-widget (`_g_*`), khôi phục khi quay lại |
| Chữ: "0.09" (đúng 0.095), README "stack 10 lớp" (mặc định 6, tối đa 10), README "mọi backward" | sửa chữ; thêm gradient check cho `Flatten` và BN eval |
| Test đủ 7 preset | test mọi control của từng preset |
| Run nặng nhất trang 12 ~7 s > ~5 s | chấp nhận, ghi vào spec: hiếm, có spinner |

## Test

- `tests/test_analysis.py`: `receptive_field` khớp 4/10/22 và công thức với
  stride/kernel khác; `receptive_field_map` đo được đúng extent đó; `pca_2d`
  trục đầu là hướng phương sai lớn nhất; `nearest_centroid_accuracy` trên dữ
  liệu tách rõ = 1.0; `occlusion_map` shape + che vùng trống không làm rơi xác
  suất; `saliency_map` khớp đạo hàm số và không đổi trọng số / cache của model
  gốc; `confusion_matrix` đếm đúng.
- `tests/test_layers.py`: BN eval backward khớp đạo hàm số; Flatten grad check;
  ReLU không đọc lại `self.mask`.
- `tests/test_train.py`: MLP build/train chạy, `count_params` đúng, arch lạ
  raise, `shift_images` dịch đúng.
- `tests/test_viz.py`: `cmap=None` chọn DIV/WARM đúng; `fig_scatter_grid` vẽ đủ
  panel.
- `tests/test_pages.py`: trang 13, 14 render; toast; khôi phục control trang
  12; đủ 7 preset; `FALSE_CLAIMS` mở rộng nếu cần.

## Ngoài phạm vi

Dataset lớn hơn, optimizer khác, loss landscape, Grad-CAM, activation
maximization, depthwise/1×1 conv.
