"""Khóa file để nhiều job chia việc an toàn trên cùng máy / cùng ổ đĩa.

Job muốn đọc chương N phải "giành" được file khóa ch000N.lock (tạo nguyên tử bằng
O_CREAT|O_EXCL). Job khác thấy khóa thì bỏ qua chương đó. Job chết giữa chừng: khóa
bị coi là hết hạn khi tiến trình giữ khóa không còn (cùng máy) hoặc quá `stale_sec`.
"""
import json
import os
import socket
import time
from typing import Optional

HOST = socket.gethostname()


def pid_alive(pid: int) -> Optional[bool]:
    """True/False nếu biết chắc; None nếu không kiểm tra được (Windows)."""
    if os.name == "nt":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def is_stale(path: str, stale_sec: float) -> bool:
    try:
        age = time.time() - os.path.getmtime(path)
    except FileNotFoundError:
        return True
    owner = _owner(path)
    if owner.get("host") == HOST and owner.get("pid") and owner.get("pid") != os.getpid():
        alive = pid_alive(int(owner["pid"]))
        if alive is False:
            return True
    return age > stale_sec


def claim(path: str, job: str = "", stale_sec: float = 3 * 3600) -> bool:
    """Giành khóa. True = được quyền làm việc này."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if is_stale(path, stale_sec):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                continue
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": HOST, "job": job, "time": time.time()}, handle)
        return True
    return False


def release(path: str) -> None:
    owner = _owner(path)
    if owner.get("pid") in (None, os.getpid()):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def touch(path: str) -> None:
    try:
        os.utime(path, None)
    except FileNotFoundError:
        pass
