"""Đẩy video lên Google Drive bằng rclone."""
import os
import shutil
import subprocess
from typing import Any, Dict

from core.utils import log


def upload(path: str, cfg: Dict[str, Any], log_path: str = "") -> bool:
    if not cfg.get("enabled", False):
        return False
    rclone = shutil.which("rclone")
    if not rclone:
        log("Không có rclone; giữ video trên máy.", "Drive")
        return False
    remote = str(cfg.get("remote_dir", "")).rstrip("/")
    if not remote:
        log("Chưa cấu hình upload.remote_dir; giữ video trên máy.", "Drive")
        return False
    cmd = [rclone, "copyto", path, f"{remote}/{os.path.basename(path)}"]
    config = cfg.get("config_path")
    if config:
        if not os.path.exists(config):
            log(f"Thiếu {config}; giữ video trên máy.", "Drive")
            return False
        cmd += ["--config", config]
    cmd += ["--retries", "5", "--low-level-retries", "20"]
    log(f"Đang đẩy {os.path.basename(path)} lên Drive...", "Drive")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"rclone lỗi (mã {result.returncode}); giữ file để thử lại lần sau.", "Drive")
        return False
    log(f"Đã đẩy xong {os.path.basename(path)}.", "Drive")
    return True
