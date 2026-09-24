# CNN Visualizer

Trực quan hoá cơ chế hoạt động của Convolutional Neural Network bằng cách tự
cài đặt lại toàn bộ phép toán lõi (convolution, padding, stride, pooling,
activation) bằng NumPy thuần — không dùng `torch.nn`, không dùng
`scipy.signal` — rồi quan sát từng bước biến đổi qua một app Streamlit
tương tác.

Bài tự học trong quá trình học CNN, đi trước `Handwritten_LeNet` trong lộ
trình — mục tiêu không phải huấn luyện model cho ra accuracy cao, mà là hiểu
*vì sao* các phép toán trong 1 lớp `nn.Conv2d`/`nn.MaxPool2d` lại cho ra đúng
con số và đúng kích thước như vậy.

## Vì sao project này tồn tại

Phần lớn tài liệu học CNN mô tả convolution bằng công thức tĩnh:

```
output = (W - K + 2P) / S + 1
```

Công thức này dễ nhớ nhầm hoặc dễ quên vì nó không gắn với hình dung cụ thể
nào. Project này đảo ngược cách tiếp cận: **tự tay viết vòng lặp trượt
kernel qua từng vị trí**, để công thức trên trở thành hệ quả hiển nhiên của
việc "đếm xem kernel trượt được bao nhiêu lần" — thay vì một thứ phải học
thuộc.

## Tính năng

App gồm 14 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học. Trang 1–8 (phần 1) là forward pass với kernel random; trang 9–12 (phần 2) train thật 1 CNN nhỏ để thấy BatchNorm, loss, backprop và overfit; trang 13–14 (phần 3) đi vào bản chất: conv khác Dense ở đâu, hơn ở đâu, và mạng thật sự nhìn thấy gì:

| # | Trang | Nội dung |
|---|---|---|
| 1 | Kernel & Convolution | Bước qua từng vị trí kernel trượt, hiện phép tính nhân-cộng cụ thể (`10×1 + 10×0 + ... = 34`), output được "lấp dần" theo từng bước |
| 2 | Padding | Thêm viền 0 quanh input, quan sát kích thước input thay đổi theo tham số `pad` |
| 3 | Stride | Bước qua từng vị trí kernel trượt với bước nhảy > 1, thấy rõ những vị trí bị bỏ qua |
| 4 | Output Shape | Công thức `(W - K + 2P) / S + 1`, nhập tay để đối chiếu, thử bộ số làm phép chia không chia hết để thấy lỗi hợp lệ |
| 5 | Pooling | So sánh trực quan max-pooling và average-pooling trên cùng 1 input |
| 6 | Activation | ReLU cắt giá trị âm trên feature map; so sánh đường cong ReLU với Sigmoid |
| 7 | Conv Block | Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh (RGB), quan sát nhiều feature map ra từ nhiều kernel, histogram giá trị sau từng bước |
| 8 | Full Pipeline | Xếp 2-3 Conv Block liên tiếp, forward 1 ảnh thật qua từng bước Conv / ReLU / Pool của từng block, theo dõi shape thu nhỏ dần và std activation co lại qua độ sâu |
| 9 | BatchNorm | Tính μ, σ², x̂, γx̂+β bằng số trên 1 mini-batch; stack tới 10 lớp conv random có/không BN; train mode vs eval mode (running stats); BN giúp train nhanh hơn |
| 10 | Softmax + Cross-Entropy | Kéo logits xem xác suất, loss `-log p` và gradient `p − y`; 1 ảnh chữ số đi hết mạng đã train tới tận loss |
| 11 | Backprop & Training | Đường học, ảnh hưởng của learning rate, gradient norm từng lớp, kernel và feature map trước vs sau khi train |
| 12 | Generalization | Overfit có chủ đích (ít data + nhãn sai), so sánh chồng dropout / weight decay / augmentation / BatchNorm / thêm data, early stopping, ảnh val bị đoán sai |
| 13 | Vì sao là CNN? | Đếm tham số conv vs Dense cùng shape; equivariance (dịch ảnh → feature map dịch theo, max pool chỉ bất biến một phần); train CNN vs MLP nhiều tham số hơn trên cùng data, đo trên ảnh gốc và ảnh dịch 1px |
| 14 | Mạng nhìn thấy gì? | Receptive field lý thuyết vs đo bằng gradient, effective RF dồn về giữa; PCA của cả tập val qua từng lớp (mạng đã train vs random); occlusion, saliency, confusion matrix |

## Kiến trúc & nguyên tắc thiết kế

Codebase tách bạch rõ phần **tự viết để học** và phần **hạ tầng UI có sẵn**:

```
cnn_core/   toàn bộ phép toán CNN — phần 1 tự cài đặt bằng vòng lặp NumPy thuần,
            phần 2 (layers.py, train.py) vectorized + có backward để train
viz/        theme, helper vẽ matplotlib, widget Streamlit dùng chung
pages/      1 file = 1 khái niệm, chỉ gọi vào cnn_core/ rồi vẽ kết quả
tests/      giá trị kỳ vọng tính tay sẵn, dùng để tự chấm đúng/sai cnn_core/
```

Convention quan trọng: convolution trong CNN là **cross-correlation** (kernel
không bị lật 180°) — đúng quy ước dùng trong deep learning, khác định nghĩa
"convolution" trong toán thuần tuý.

## Công nghệ sử dụng

| Thư viện | Vai trò |
|---|---|
| [NumPy](https://numpy.org/) | Toàn bộ phép toán CNN trong `cnn_core/` — chỉ dùng cho lưu trữ mảng và các phép element-wise cơ bản (`np.sum`, `np.max`, `np.mean`, `np.maximum`, `np.exp`), không dùng bất kỳ hàm convolution/pooling dựng sẵn nào |
| [Streamlit](https://streamlit.io/) | Giao diện tương tác — multipage app, widget (slider, data editor, step control), state quản lý qua `st.session_state` |
| [Matplotlib](https://matplotlib.org/) | Vẽ heatmap ma trận, highlight vị trí kernel, lưới feature map |
| [Pandas](https://pandas.pydata.org/) | Backend cho `st.data_editor` — bảng số cho phép sửa tay từng ô input/kernel |
| [Pillow](https://python-pillow.org/) | Đọc và resize ảnh người dùng upload ở trang Full Pipeline |
| [scikit-learn](https://scikit-learn.org/) | Chỉ để lấy dataset `load_digits` (1797 ảnh chữ số 8×8, có sẵn offline) cho trang 9–12 |
| [pytest](https://pytest.org/) | Test suite với giá trị kỳ vọng tính tay, dùng để tự chấm đúng/sai khi cài đặt `cnn_core/` |

## Cài đặt

```bash
git clone <repo-url>
cd CNN_Visualizer
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách sử dụng

**Chạy app:**

```bash
streamlit run app.py
```

Trình duyệt tự mở tại `http://localhost:8501`. Trang chủ hiển thị tiến độ
implement từng nhóm hàm trong `cnn_core/`, bấm vào từng card để mở trang
tương ứng. Các trang trong `pages/` cũng xuất hiện ở sidebar bên trái.

**Chạy test:**

```bash
pytest -v
```

Mỗi file trong `tests/` chứa giá trị kỳ vọng **tính tay sẵn** (không phụ
thuộc `torch`/`scipy`) để tự chấm đúng/sai khi cài đặt lại `cnn_core/`.

## Cấu trúc thư mục

```
CNN_Visualizer/
├── app.py                         # Landing page — tiến độ implement + điều hướng
├── pages/                         # 1 file = 1 khái niệm (xem bảng Tính năng)
├── cnn_core/
│   ├── padding.py                 # pad_matrix — zero-padding quanh ma trận 2D
│   ├── convolution.py             # conv2d, conv2d_multichannel — trượt kernel, nhân-cộng
│   ├── shapes.py                  # conv_output_shape — công thức (W-K+2P)/S+1
│   ├── pooling.py                 # max_pool2d, avg_pool2d
│   ├── activation.py              # relu, sigmoid
│   ├── block.py                   # ConvBlock — Conv → ReLU → Pool, make_random_kernels
│   ├── layers.py                  # Conv2D, BatchNorm2D, Dropout, Dense... có forward + backward (vectorized)
│   ├── train.py                   # dataset digits, build_model (cnn | mlp), SGD momentum + weight decay, train()
│   └── analysis.py                # receptive field, PCA, nearest-centroid, occlusion, saliency, confusion matrix
├── viz/
│   ├── theme.py                   # bảng màu, hero header, card, step-progress dots
│   ├── matrix_view.py             # heatmap ma trận, highlight cửa sổ, lưới feature map
│   ├── sliding_window.py          # liệt kê toạ độ mọi vị trí kernel trượt qua
│   └── widgets.py                 # editable_matrix, step_controls, run_or_hint
├── tests/                         # giá trị kỳ vọng tính tay cho từng hàm cnn_core/
├── data/sample_images/            # nơi đặt ảnh mẫu cho trang Full Pipeline (tuỳ chọn)
└── TODO.md                        # lộ trình implement theo từng phase
```

## Chi tiết cài đặt các phép toán

### Convolution (`cnn_core/convolution.py`)

`conv2d(input, kernel, stride, padding)` cài đặt trực tiếp định nghĩa toán
học của convolution 2D dạng rời rạc: pad input, sau đó với mỗi vị trí output
`(i, j)`, cắt ra cửa sổ input tương ứng và tính tích trong (element-wise
multiply rồi sum) với kernel:

```
output[i, j] = Σ_u Σ_v  input_padded[i·S + u, j·S + v] · kernel[u, v]
```

`conv2d_multichannel` mở rộng sang input nhiều kênh (`H, W, C_in`) và nhiều
kernel (`K, K, C_in, C_out`): với mỗi kernel, convolve riêng từng kênh input
rồi **cộng** kết quả của `C_in` kênh lại thành 1 feature map duy nhất — đúng
cách một lớp `nn.Conv2d` xử lý ảnh RGB.

### Padding (`cnn_core/padding.py`)

`pad_matrix` tạo một ma trận toàn số 0 kích thước `(H + 2·pad, W + 2·pad)`
rồi gán ma trận gốc vào vùng giữa bằng slicing — không dùng `np.pad`.

### Stride

Không phải hàm riêng — là tham số của `conv2d`/pooling quyết định bước nhảy
của vòng lặp trượt cửa sổ (`i · stride`, `j · stride`), tách biệt khỏi kích
thước kernel.

### Output shape (`cnn_core/shapes.py`)

`conv_output_shape` tính số vị trí hợp lệ mà kernel có thể trượt tới trên
1 chiều, và raise `ValueError` khi bộ tham số không hợp lệ (kernel "hụt
chân" ở cuối input, tức phép chia không chia hết).

### Pooling (`cnn_core/pooling.py`)

`max_pool2d`/`avg_pool2d` dùng cùng cấu trúc vòng lặp trượt cửa sổ như
`conv2d`, nhưng không có trọng số — mỗi cửa sổ được tóm tắt bằng `np.max`
hoặc `np.mean`.

### Activation (`cnn_core/activation.py`)

`relu(x) = max(0, x)` và `sigmoid(x) = 1 / (1 + e^-x)`, áp dụng
element-wise trên toàn bộ feature map.

### ConvBlock (`cnn_core/block.py`)

`ConvBlock.forward` ghép 3 hàm trên theo đúng thứ tự chuẩn của 1 khối CNN cổ
điển: `conv2d_multichannel → relu → max_pool2d` (áp dụng riêng từng kênh vì
`max_pool2d` chỉ nhận input 2D). Kernel được sinh ngẫu nhiên
(`make_random_kernels`, phân phối `N(0, 0.1)`) — phần 1 không train, mục tiêu
chỉ là quan sát forward pass, tương đương nhìn một model ngay sau khi khởi
tạo (training nằm ở phần 2). `forward_steps` trả về kết quả sau từng
bước để trang Full Pipeline đi qua được Conv / ReLU / Pool riêng lẻ.

### Layers có backward (`cnn_core/layers.py`)

Phần 2 cần train, nên vòng lặp tay quá chậm. Mỗi lớp (`Conv2D`, `ReLU`,
`MaxPool2D`, `BatchNorm2D`, `Dropout`, `Flatten`, `Dense`) có `forward(x,
train)` và `backward(dout)`, nhận batch `(N, H, W, C)`. Conv và pool xếp mọi
cửa sổ `k×k` ra 1 mảng (`_windows`, tương đương im2col) rồi tính bằng 1 phép
matmul/max; backward cộng dồn gradient ngược về từng pixel gốc
(`_windows_backward`). `BatchNorm2D` dùng thống kê của batch khi train và
running stats khi eval. `softmax_cross_entropy_backward` trả về `(p − y)/N`.

Mọi backward (trừ `Dropout`, kiểm bằng test hành vi) được kiểm bằng gradient
check (sai phân trung tâm) — `BatchNorm2D` ở cả train mode lẫn eval mode. `Conv2D`
được đối chiếu với `conv2d_multichannel` vòng lặp tay, và (nếu có `torch`)
với `torch.nn.functional.conv2d` / `batch_norm`.

### Training (`cnn_core/train.py`)

Dataset `load_digits` (8×8, 10 lớp), 500 ảnh val cố định. Mô hình:
`Conv(16) → [BN] → ReLU → Pool → Conv(32) → [BN] → ReLU → Pool → Dense(128→64)
→ ReLU → [Dropout] → Dense(64→10)` — cố ý dư sức chứa để overfit hiện rõ.
SGD momentum 0.9, weight decay chỉ áp cho `W`. Tuỳ chọn `label_noise` đổi 1 tỉ lệ
ảnh train sang nhãn sai ngẫu nhiên để mô phỏng dữ liệu bẩn. Train full data
~1 giây, val acc ~99%. `arch="mlp"` dựng MLP 24k tham số (nhiều hơn CNN) để
so sánh ở trang 13.

### Phân tích mô hình (`cnn_core/analysis.py`)

`receptive_field` tính vùng nhìn lý thuyết (`r ← r + (k − 1)·jump`,
`jump ← jump·stride`); `receptive_field_map` đo nó bằng gradient của 1 neuron
giữa về input. `pca_2d` + `nearest_centroid_accuracy` đo các lớp chữ số tách
nhau thế nào ở từng lớp. `occlusion_map` (che từng ô, xem xác suất rơi) và
`saliency_map` (|∂logit/∂pixel|, chạy trên bản sao model để không đụng model
đang cache) chỉ ra pixel nào quyết định dự đoán.

## Kiểm thử

Test phần 1 dùng giá trị kỳ vọng tính tay trên các ma trận nhỏ (4×4, 2×2) —
không dùng hàm nào từ `cnn_core/` để sinh ra "đáp án", tránh trường hợp test
tự khớp với chính lỗi của cài đặt. Test phần 2 dùng gradient check số học và
đối chiếu với bản vòng lặp tay / torch. `tests/test_pages.py` chạy thử mọi
trang Streamlit bằng `AppTest`.

```bash
pytest -v
```

## Trạng thái hoàn thành

Toàn bộ 6 phase trong `TODO.md` đã hoàn tất, cả 14 trang Streamlit chạy
được — từ input tuỳ chỉnh, qua ảnh thật đi qua nhiều Conv Block, tới train
thật, so sánh các kỹ thuật chống overfit, và soi xem mạng nhìn thấy gì.

## Định hướng mở rộng

- Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur để xây trực giác
  "kernel = bộ dò 1 loại pattern"
- Padding `mode="reflect"` bên cạnh zero-padding, so sánh viền ảnh
- Benchmark tốc độ: vòng lặp tay vs `np.einsum`/vectorize hoá — thấy vì sao
  framework thật không dùng for-loop thuần Python
