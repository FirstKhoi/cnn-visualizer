"""Widget dùng chung cho các trang pages/*.py — đã viết sẵn hoàn chỉnh."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st


def editable_matrix(key: str, matrix: np.ndarray) -> np.ndarray:
    """Bảng số cho phép sửa trực tiếp trên UI (vd: chỉnh tay từng ô kernel)."""
    df = pd.DataFrame(matrix)
    edited = st.data_editor(df, key=key, num_rows="fixed")
    return edited.to_numpy(dtype=float)


def run_or_hint(fn: Callable, *args: Any, todo_hint: str, **kwargs: Any):
    """Gọi fn(*args, **kwargs); nếu chưa implement (NotImplementedError) thì
    hiện gợi ý thay vì traceback đỏ lòm, rồi dừng render phần còn lại của
    trang (vì không có kết quả để vẽ tiếp)."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        st.info(todo_hint)
        st.stop()
    except Exception as exc:
        st.error(f"Lỗi khi chạy `{fn.__name__}`: {exc}")
        st.stop()
