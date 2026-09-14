# CNN Visualizer

App Streamlit để **tự tay code** các phép toán lõi của CNN (kernel, padding,
stride, pooling, activation) rồi xem trực quan chúng biến đổi ma trận/ảnh ra
sao — thay vì chỉ gọi `torch.nn.Conv2d` như ở `Handwritten_LeNet`.

## Mục tiêu học

- Hiểu convolution 2D là phép nhân-cộng trượt cửa sổ, không phải hộp đen.
- Thấy bằng mắt padding/stride ảnh hưởng kích thước output thế nào, thay vì
  chỉ nhớ công thức `(W - K + 2P) / S + 1`.
- Ghép các phép toán rời rạc thành 1 `ConvBlock` (Conv → ReLU → Pool), rồi
  xếp nhiều block để thấy feature map co nhỏ dần, giống hệt cách LeNet hoạt
  động — nhưng lần này tự viết forward pass, không dùng framework.

## Nguyên tắc của project này

`cnn_core/` (convolution, padding, pooling, activation, shapes, block) **cố
tình để trống** dạng hàm rỗng + docstring/hint — đó là phần bạn tự viết để
hiểu. `viz/` và `pages/` đã viết sẵn UI (Streamlit + matplotlib), chỉ gọi vào
`cnn_core` — implement xong 1 hàm là thấy kết quả ngay trên trang tương ứng.

Xem lộ trình chi tiết từng bước trong [`TODO.md`](TODO.md).

## Cài đặt

```bash
cd CNN_Visualizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Cách chạy

```bash
streamlit run app.py
```

Tự chấm đúng/sai khi implement `cnn_core/`:

```bash
pytest -v
```

## Cấu trúc

```
CNN_Visualizer/
├── app.py                  # Trang chủ Streamlit — tổng quan + điều hướng
├── pages/                  # Mỗi trang = 1 khái niệm, theo đúng thứ tự học trong TODO.md
├── cnn_core/                # ← Bạn viết phần này (TODO, không dùng torch/scipy)
│   ├── padding.py
│   ├── convolution.py
│   ├── pooling.py
│   ├── activation.py
│   ├── shapes.py
│   └── block.py
├── viz/                    # Helper vẽ ma trận/heatmap/highlight — đã viết sẵn
├── data/sample_images/     # Ảnh mẫu cho page Full Pipeline
└── tests/                  # pytest — giá trị tính tay sẵn để tự chấm cnn_core/
```

## Kết quả đo được

_(cập nhật sau khi implement xong — vd: ảnh MNIST đi qua 2 block tự viết cho
feature map khớp visually với feature map của `torch.nn.Conv2d` cùng kernel)._
