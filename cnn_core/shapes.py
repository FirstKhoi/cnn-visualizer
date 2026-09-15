"""Phase 3 — Công thức output shape.

Một khi hiểu conv2d bằng vòng lặp, công thức (W - K + 2P) / S + 1 không còn
là thứ phải học thuộc — nó chỉ là đếm xem kernel trượt được bao nhiêu lần.
"""


def conv_output_shape(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """Tính kích thước 1 chiều (H hoặc W) của output sau convolution/pooling.

    Args:
        input_size: kích thước input theo chiều đang xét (H hoặc W).
        kernel_size: kích thước kernel/window theo chiều đó.
        stride: bước nhảy.
        padding: pad thêm vào MỖI cạnh của chiều đó.

    Returns:
        Kích thước output (int).

    Raises:
        ValueError: nếu (input_size + 2*padding - kernel_size) không chia
            hết cho stride — nghĩa là bộ tham số này không hợp lệ (kernel sẽ
            "hụt chân" ở cuối input). Thông báo lỗi nên nêu rõ input_size,
            kernel_size, stride, padding để dễ debug trên UI.

    Hint:
        output = (input_size + 2*padding - kernel_size) // stride + 1
        Kiểm tra chia hết bằng (input_size + 2*padding - kernel_size) % stride == 0
        trước khi return.
    """
    if stride <= 0:
        raise ValueError(f'Stride must be postive, got stride={stride}')
    
    numerator =input_size + 2 * padding - kernel_size
    if numerator < 0:
        raise ValueError(
            f"Kernel size ({kernel_size}) is larger than padded input size "
            f"({input_size + 2 * padding}). Params: input_size={input_size}, "
            f"kernel_size={kernel_size}, stride={stride}, padding={padding}."
        )
    if numerator % stride != 0:
        raise ValueError(
            f"Invalid dimensions: (input_size + 2*padding - kernel_size) = {numerator} "
            f"is not divisible by stride={stride}. "
            f"Params: input_size={input_size}, kernel_size={kernel_size}, "
            f"stride={stride}, padding={padding}."
        )

    return numerator // stride + 1
