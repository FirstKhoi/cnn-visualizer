"""Widget dùng chung cho các trang pages/*.py — đã viết sẵn hoàn chỉnh."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from viz.theme import PRIMARY, step_dots


def editable_matrix(key: str, matrix: np.ndarray) -> np.ndarray:
    """Bảng số cho phép sửa trực tiếp trên UI (vd: chỉnh tay từng ô kernel)."""
    df = pd.DataFrame(matrix)
    edited = st.data_editor(df, key=key, num_rows="fixed")
    return edited.to_numpy(dtype=float)


def step_controls(key: str, n_steps: int, color: str = PRIMARY, step_label: str = "Bước") -> int:
    """Bộ điều khiển 'Trước / Tiếp' + chấm tiến trình, quản lý vị trí bước
    hiện tại qua st.session_state. Dùng cho các trang mô phỏng step-by-step
    (kernel trượt qua từng vị trí, pipeline chạy qua từng block, ...).

    Trả về index bước hiện tại (0-based), luôn nằm trong [0, n_steps-1].
    """
    state_key = f"__step__{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    current = min(st.session_state[state_key], n_steps - 1)

    col_prev, col_label, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Trước", key=f"{key}_prev", disabled=(current == 0), use_container_width=True):
            current = max(0, current - 1)
    with col_next:
        if st.button("Tiếp ▶", key=f"{key}_next", disabled=(current == n_steps - 1), use_container_width=True):
            current = min(n_steps - 1, current + 1)
    with col_label:
        st.markdown(
            f'<div style="text-align:center; font-weight:700; color:{color}; padding-top:0.4rem;">'
            f"{step_label} {current + 1}/{n_steps}</div>",
            unsafe_allow_html=True,
        )

    st.session_state[state_key] = current
    step_dots(current, n_steps, color=color)
    return current


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
