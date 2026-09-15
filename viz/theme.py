"""Theme "Vibrant Learning" — màu sắc, CSS, và các component UI dùng chung
cho mọi trang. Tất cả đã viết sẵn hoàn chỉnh, không phải phần bạn cần
implement.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PRIMARY = "#6C5CE7"       # tím-indigo — màu chủ đạo
PRIMARY_DARK = "#5847D6"
ACCENT = "#FDCB6E"        # cam/vàng ấm — điểm nhấn
BG = "#F8F9FC"
CARD_BG = "#FFFFFF"
TEXT = "#2D3436"
MUTED = "#636E72"
BORDER = "#E9E6FB"
SUCCESS = "#00B894"

# Mỗi trang 1 màu riêng để dễ nhận diện đang ở đâu trong hành trình học
PAGE_COLORS = {
    1: "#6C5CE7",  # Kernel & Convolution — tím
    2: "#4D96FF",  # Padding — xanh dương
    3: "#00B894",  # Stride — xanh lá/ngọc
    4: "#FDCB6E",  # Output shape — vàng cam
    5: "#FF7675",  # Pooling — san hô
    6: "#A29BFE",  # Activation — lavender
    7: "#FD79A8",  # Conv Block — hồng
    8: "#E17055",  # Full Pipeline — cam đất
}


def inject_base_css() -> None:
    """Gọi 1 lần ở đầu mỗi trang — CSS nền, card, button, sidebar dùng chung."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {BG};
        }}
        .block-container {{
            padding-top: 2rem;
            max-width: 1100px;
        }}
        h1, h2, h3 {{
            color: {TEXT};
            font-weight: 700;
        }}
        p, li, label, .stMarkdown {{
            color: {TEXT};
        }}
        [data-testid="stSidebar"] {{
            background: {CARD_BG};
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] * {{
            color: {TEXT};
        }}
        .stButton > button {{
            border-radius: 10px;
            border: 1.5px solid {PRIMARY};
            color: {PRIMARY};
            background: {CARD_BG};
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background: {PRIMARY};
            color: white;
            border-color: {PRIMARY};
        }}
        .stButton > button:disabled {{
            opacity: 0.35;
        }}
        [data-testid="stMetricValue"] {{
            color: {PRIMARY};
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            background: {CARD_BG};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
        }}
        code {{
            color: {PRIMARY_DARK};
            background: {BORDER};
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badge: str, color: str = PRIMARY) -> None:
    """Header gradient màu {color} -> ACCENT, dùng ở đầu mỗi trang."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(120deg, {color} 0%, {ACCENT} 100%);
            border-radius: 18px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px -8px {color}66;
        ">
            <div style="
                display:inline-block;
                background: rgba(255,255,255,0.25);
                color: white;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                padding: 0.2rem 0.7rem;
                border-radius: 999px;
                margin-bottom: 0.6rem;
            ">{badge}</div>
            <div style="color: white; font-size: 1.9rem; font-weight: 800; line-height: 1.2;">
                {title}
            </div>
            <div style="color: rgba(255,255,255,0.92); font-size: 1.02rem; margin-top: 0.35rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, color: str = PRIMARY, label: str = "Vì sao quan trọng") -> None:
    """Ô ghi chú màu, dùng để giải thích ý nghĩa/trực giác của 1 khái niệm."""
    st.markdown(
        f"""
        <div style="
            background: {color}14;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.75rem 0;
            color: {TEXT};
        ">
            <div style="font-weight:700; color:{color}; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.03em;">
                {label}
            </div>
            <div style="margin-top:0.2rem;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def formula_box(formula_html: str, color: str = PRIMARY) -> None:
    """Ô hiển thị phép tính hiện tại (vd: '3*1 + 4*0 + ... = 14') — dùng ở
    các bước step-by-step để lộ ra đúng con số đang được tính."""
    st.markdown(
        f"""
        <div style="
            background: {TEXT};
            color: {ACCENT};
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 1.05rem;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin: 0.6rem 0;
            border: 1px solid {color};
            overflow-x: auto;
            white-space: nowrap;
        ">
            {formula_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_dots(current: int, total: int, color: str = PRIMARY) -> None:
    """Hàng chấm tròn thể hiện đang ở bước nào trong tổng số bước."""
    dots = "".join(
        f'<span style="display:inline-block; width:{10 if i == current else 7}px; '
        f'height:{10 if i == current else 7}px; border-radius:50%; margin:0 4px; '
        f'background:{color if i <= current else BORDER}; '
        f'transition: all 0.15s ease;"></span>'
        for i in range(total)
    )
    st.markdown(f'<div style="margin: 0.4rem 0 1rem 0;">{dots}</div>', unsafe_allow_html=True)


def phase_pill(text: str, color: str) -> str:
    """Trả về HTML 1 pill nhỏ màu — dùng chèn trong markdown khác (vd bảng trạng thái)."""
    return (
        f'<span style="background:{color}22; color:{color}; font-weight:700; '
        f'font-size:0.75rem; padding:0.15rem 0.6rem; border-radius:999px;">{text}</span>'
    )
