"""Google Drive qua rclone: tải chương/tài nguyên về, đẩy video lên, xem tập nào đã có."""
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set

from core.utils import log


def _base_cmd(config_path: Optional[str]) -> Optional[List[str]]:
    rclone = shutil.which("rclone")
    if not rclone:
        log("Không có rclone.", "Drive")
        return None
    cmd = [rclone]
    if config_path:
        if not os.path.exists(config_path):
            log(f"Thiếu {config_path}.", "Drive")
            return None
        cmd += ["--config", config_path]
    return cmd


def upload(path: str, cfg: Dict[str, Any], log_path: str = "") -> bool:
    if not cfg.get("enabled", False):
        return False
    remote = str(cfg.get("remote_dir", "")).rstrip("/")
    if not remote:
        log("Chưa cấu hình upload.remote_dir; giữ video trên máy.", "Drive")
        return False
    base = _base_cmd(cfg.get("config_path"))
    if base is None:
        log("Không upload được; giữ video trên máy.", "Drive")
        return False
    cmd = base + ["copyto", path, f"{remote}/{os.path.basename(path)}", "--retries", "5", "--low-level-retries", "20"]
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


def pull(remote: str, local: str, config_path: Optional[str], includes: Optional[List[str]] = None) -> bool:
    """Tải thư mục Drive về máy; file đã có và không đổi thì bỏ qua (chỉ lấy file mới)."""
    base = _base_cmd(config_path)
    if base is None or not remote:
        return False
    os.makedirs(local, exist_ok=True)
    cmd = base + ["copy", remote.rstrip("/"), local, "--retries", "5", "--low-level-retries", "20"]
    for pattern in includes or []:
        cmd += ["--include", pattern]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        log(f"Không tải được {remote}: {result.stderr.strip()[-300:]}", "Drive")
        return False
    return True


def list_files(remote: str, config_path: Optional[str]) -> Optional[List[str]]:
    """Tên các file trong thư mục Drive (None nếu không đọc được)."""
    base = _base_cmd(config_path)
    if base is None or not remote:
        return None
    result = subprocess.run(base + ["lsf", remote.rstrip("/"), "--files-only"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        # Thư mục chưa tồn tại (chưa upload tập nào) không phải lỗi.
        if "directory not found" in result.stderr.lower():
            return []
        log(f"Không đọc được {remote}: {result.stderr.strip()[-300:]}", "Drive")
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_dirs(remote: str, config_path: Optional[str]) -> Optional[List[str]]:
    """Tên các thư mục con trong thư mục Drive (None nếu không đọc được)."""
    base = _base_cmd(config_path)
    if base is None or not remote:
        return None
    result = subprocess.run(base + ["lsf", remote.rstrip("/"), "--dirs-only"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    return [line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()]


def fold_name(name: str) -> str:
    """'Đại Niết Bàn' / 'dai_niet_ban' / 'Dai  Niet Ban' -> 'dai niet ban' (so khớp tên không dấu)."""
    import unicodedata

    text = unicodedata.normalize("NFD", name.replace("Đ", "D").replace("đ", "d"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[\s_\-]+", " ", text.lower())
    return text.strip()


def match_folder(wanted: str, names: List[str]) -> Optional[str]:
    """Tên thư mục trên Drive khớp với tên gõ vào (đúng y hệt, hoặc khớp khi bỏ dấu)."""
    if wanted in names:
        return wanted
    hits = [n for n in names if fold_name(n) == fold_name(wanted)]
    return hits[0] if len(hits) == 1 else None


EPISODE_RE = re.compile(r"^Chuong_(\d{4,})-(\d{4,})_Tap(\d{3,})\.mp4$")


def uploaded_episodes(names: List[str]) -> Set[str]:
    """Tên file video tập (Chuong_0001-0010_Tap001.mp4) đã có trên Drive."""
    return {name for name in names if EPISODE_RE.match(name)}
