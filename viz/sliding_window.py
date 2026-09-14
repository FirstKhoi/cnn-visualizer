"""Helper tính vị trí cửa sổ trượt cho slider "step qua từng bước" trên UI.
Chỉ là phép đếm vị trí (row, col) — không phải phép toán convolution, nên
viết sẵn ở đây thay vì bắt implement lại trong cnn_core/.
"""

from __future__ import annotations


def window_positions(input_size: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Liệt kê toạ độ (row, col) góc trên-trái của mọi vị trí cửa sổ hợp lệ,
    theo đúng thứ tự conv2d/pooling sẽ duyệt qua (trái->phải, trên->dưới).

    Ví dụ: input_size=4, window_size=2, stride=2 -> [(0,0), (0,2), (2,0), (2,2)]
    """
    positions = []
    row = 0
    while row + window_size <= input_size:
        col = 0
        while col + window_size <= input_size:
            positions.append((row, col))
            col += stride
        row += stride
    return positions
