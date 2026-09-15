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

App gồm 8 trang, mỗi trang tương ứng đúng 1 khái niệm, đi theo thứ tự nên học:

| # | Trang | Nội dung |
|---|---|---|
| 1 | Kernel & Convolution | Bước qua từng vị trí kernel trượt, hiện phép tính nhân-cộng cụ thể (`10×1 + 10×0 + ... = 34`), output được "lấp dần" theo từng bước |
| 2 | Padding | Thêm viền 0 quanh input, quan sát kích thước input thay đổi theo tham số `pad` |
| 3 | Stride | Bước qua từng vị trí kernel trượt với bước nhảy > 1, thấy rõ những vị trí bị bỏ qua |
| 4 | Output Shape | Công thức `(W - K + 2P) / S + 1`, nhập tay để đối chiếu, thử bộ số làm phép chia không chia hết để thấy lỗi hợp lệ |
| 5 | Pooling | So sánh trực quan max-pooling và average-pooling trên cùng 1 input |
| 6 | Activation | ReLU cắt giá trị âm trên feature map; so sánh đường cong ReLU với Sigmoid |
| 7 | Conv Block | Ghép Conv → ReLU → Pool thành 1 khối, chạy trên ảnh nhiều kênh (RGB), quan sát nhiều feature map ra từ nhiều kernel |
| 8 | Full Pipeline | Xếp 2-3 Conv Block liên tiếp, forward 1 ảnh thật (upload hoặc ảnh tổng hợp) qua từng block một, theo dõi shape thu nhỏ dần |

## Kiến trúc & nguyên tắc thiết kế

Codebase tách bạch rõ phần **tự viết để học** và phần **hạ tầng UI có sẵn**:

```
cnn_core/   toàn bộ phép toán CNN — tự cài đặt bằng vòng lặp NumPy thuần
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
│   └── block.py                   # ConvBlock — Conv → ReLU → Pool, make_random_kernels
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
(`make_random_kernels`, phân phối `N(0, 0.1)`) — project này không cài đặt
backprop/training, mục tiêu chỉ là quan sát forward pass, tương đương nhìn
một model ngay sau khi khởi tạo.

## Kiểm thử

20 test case trong `tests/`, giá trị kỳ vọng được tính tay trên các ma trận
nhỏ (4×4, 2×2) trước khi viết test — không dùng bất kỳ hàm nào từ
`cnn_core/` để sinh ra "đáp án", tránh trường hợp test tự khớp với chính lỗi
của cài đặt.

```bash
pytest -v
```

## Trạng thái hoàn thành

Toàn bộ 6 phase trong `TODO.md` đã hoàn tất — 20/20 test pass, cả 8 trang
Streamlit chạy được từ input tuỳ chỉnh tới ảnh thật qua nhiều lớp Conv Block.

## Định hướng mở rộng

- So sánh output `conv2d` tự viết với `torch.nn.functional.conv2d` cùng
  kernel — số phải khớp tuyệt đối (trong sai số dấu phẩy động)
- Thêm preset kernel Sobel X/Y, Sharpen, Gaussian blur để xây trực giác
  "kernel = bộ dò 1 loại pattern"
- Padding `mode="reflect"` bên cạnh zero-padding, so sánh viền ảnh
- Benchmark tốc độ: vòng lặp tay vs `np.einsum`/vectorize hoá — thấy vì sao
  framework thật không dùng for-loop thuần Python
