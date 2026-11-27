"""Làm sạch file chương truyện trước khi đưa cho VieNeu.

Chỉ dọn phần KHÔNG phải lời truyện (bảng danh xưng, dòng nguồn/tác giả, dấu phân cách) và
thay từ theo glossary.json của bộ truyện. Chuẩn hóa số/viết tắt, tách câu, nhịp nghỉ đều do
VieNeu (sea-g2p) làm — xem core/vieneu_engine.py.
"""
import re
import unicodedata
from typing import Dict, List, Optional


def extract_standardization_table(raw_text: str) -> List[Dict[str, str]]:
    """Đọc bảng chuẩn hóa danh xưng ở cuối chương (nếu có): tên riêng để ASR không tính là sai."""
    keyword = "Bảng chuẩn hóa danh xưng"
    if keyword not in raw_text:
        return []
    items = []
    for line in raw_text.split(keyword)[-1].strip().split("\n"):
        line = line.strip().lstrip("-*•").strip()
        if ":" in line:
            name, desc = (part.strip() for part in line.split(":", 1))
            if name and len(name) < 50:
                items.append({"name": name, "description": desc})
    return items


def clean_paragraphs(raw_text: str) -> List[str]:
    """Các đoạn lời truyện (mỗi dòng không rỗng là một đoạn)."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    for keyword in (
        "Bảng chuẩn hóa danh xưng nhân vật & địa danh trong chương:",
        "Bảng chuẩn hóa danh xưng",
        "--- HẾT CHƯƠNG ---",
    ):
        if keyword in text:
            text = text.split(keyword)[0]

    paragraphs: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped in {"***", "* * *", "---"}:
            continue
        if re.match(r"^(NGUYÊN TÁC|Thời gian|Tác giả|Nguồn|Converter|Editor)\s*:", stripped, re.IGNORECASE):
            continue
        line = re.sub(r"[·•・]", "", stripped)            # c·hết -> chết
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if re.search(r"\w", line):
            paragraphs.append(line)
    return paragraphs


def apply_glossary(text: str, glossary: Optional[Dict[str, str]]) -> str:
    """Thay từ theo glossary.json của bộ truyện (cách đọc tên riêng, từ mượn...). Khóa dài thắng trước."""
    text = unicodedata.normalize("NFC", text or "")
    for key in sorted(glossary or {}, key=len, reverse=True):
        value = glossary[key]
        if not key:
            continue
        if re.fullmatch(r"[\w\s]+", key):
            text = re.sub(rf"(?<!\w){re.escape(key)}(?!\w)", value, text)
        else:
            text = text.replace(key, value)
    return text


def chapter_text(raw_text: str, glossary: Optional[Dict[str, str]] = None) -> str:
    """File chương -> văn bản đưa vào VieNeu: mỗi đoạn một dòng (VieNeu nghỉ 0,7 s giữa các đoạn)."""
    return "\n".join(apply_glossary(p, glossary) for p in clean_paragraphs(raw_text))
