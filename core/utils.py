"""Tiện ích dùng chung: log, JSON an toàn, fingerprint, đọc cấu hình."""
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional

_LOG_HANDLE = None


def set_log_file(path: Optional[str]) -> None:
    global _LOG_HANDLE
    if _LOG_HANDLE:
        _LOG_HANDLE.close()
        _LOG_HANDLE = None
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _LOG_HANDLE = open(path, "a", encoding="utf-8")


def log(message: str, tag: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}]{'[' + tag + ']' if tag else ''} {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((line + "\n").encode("utf-8", "replace"))
        sys.stdout.flush()
    if _LOG_HANDLE:
        _LOG_HANDLE.write(line + "\n")
        _LOG_HANDLE.flush()


def fingerprint(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: Optional[str]) -> Optional[str]:
    """Fingerprint nhanh theo đường dẫn + kích thước + thời gian sửa (file video nền rất lớn)."""
    if not path or not os.path.exists(path):
        return None
    stat = os.stat(path)
    return fingerprint(os.path.abspath(path), stat.st_size, int(stat.st_mtime))


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        log(f"Không đọc được {path}: {exc}", "Cảnh báo")
        return default


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def chapter_number(path: str) -> int:
    """Số chương lấy từ tên file: Chuong_0012.txt -> 12."""
    match = re.search(r"\d+", os.path.basename(path))
    return int(match.group()) if match else 999999


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
