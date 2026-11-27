#!/usr/bin/env python3
"""Audio Studio: đọc truyện thành audio mượt, ghép tập với video nền lặp, đẩy lên Drive.

  python studio.py run                 đọc audio + tự ghép tập khi đủ chương (chạy song song)
  python studio.py run --range 1-400   chỉ xử lý chương 1-400
  python studio.py audio --project X   chỉ đọc audio của bộ truyện X (chạy nhiều job cùng lúc được)
  python studio.py video --watch       job ghép video chạy ngầm: đủ 20 chương là ghép + upload
  python studio.py video [--finish]    ghép các tập đã đủ audio (--finish: gom cả tập lẻ cuối)
  python studio.py check               kiểm tra môi trường (ffmpeg, GPU, giọng mẫu, video nền)
  python studio.py bench --price 0.69  đo tốc độ + đánh giá ASR trên chương thật, quy ra chi phí
"""
import argparse
import glob
import json
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# Thư mục dữ liệu (chương, giọng mẫu, video nền, output). Trên pod đặt ở volume /workspace
# để Stop pod không mất; máy cá nhân mặc định là chính thư mục tool.
DATA = os.path.abspath(os.environ.get("AUDIO_STUDIO_DATA", ROOT))
# Thư mục gốc dữ liệu (dùng chung cho mọi bộ truyện: rclone.conf, giọng mẫu chung...).
# Khi chạy --project X thì DATA = <BASE>/projects/X.
BASE = DATA

from core import drive, locks  # noqa: E402
from core.asr_checker import ASRChecker  # noqa: E402
from core.chapter_builder import ChapterBuilder  # noqa: E402
from core.text_processor import chapter_text, extract_standardization_table  # noqa: E402
from core.audio_utils import write_wav  # noqa: E402
from core.vieneu_engine import VieneuEngine, chunk_text, join_chunks  # noqa: E402
from core.utils import (  # noqa: E402
    chapter_number, deep_merge, file_fingerprint, fingerprint, format_duration, load_config, log, read_json,
    set_log_file, write_json,
)
from core.video_maker import VideoMaker  # noqa: E402

# Đổi cách chọn đoạn giọng mẫu thì tăng số này: các pod sẽ chọn lại thay vì dùng bản cũ trong work/voice.
VOICE_PICK_VERSION = 1


def rel(path: str) -> str:
    """Đường dẫn trong thư mục bộ truyện; không có thì dùng bản chung ở thư mục gốc dữ liệu."""
    if os.path.isabs(path):
        return path
    own = os.path.join(DATA, path)
    if not os.path.exists(own) and DATA != BASE and os.path.exists(os.path.join(BASE, path)):
        return os.path.join(BASE, path)
    return own


def own(path: str) -> str:
    """Đường dẫn RIÊNG của bộ truyện (không quay về thư mục chung): dùng cho dữ liệu ghi ra."""
    return path if os.path.isabs(path) else os.path.join(DATA, path)


def fill_project(value: Any, project: str) -> Any:
    """Thay {project} trong mọi chuỗi cấu hình bằng tên bộ truyện."""
    if isinstance(value, str):
        return value.replace("{project}", project)
    if isinstance(value, dict):
        return {fill_project(k, project): fill_project(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_project(v, project) for v in value]
    return value


def find_rclone_conf(cfg: Dict[str, Any]) -> Optional[str]:
    """rclone.conf: theo config -> thư mục dữ liệu chung -> mặc định của rclone (~/.config/rclone)."""
    name = cfg.get("upload", {}).get("config_path") or "rclone.conf"
    candidates = [rel(name), os.path.join(BASE, name),
                  os.path.join(os.path.expanduser("~"), ".config", "rclone", "rclone.conf")]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def resolve_auto(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """sync/upload 'enabled: auto' -> bật khi tìm thấy rclone.conf (RunPod Secret hoặc file)."""
    has_conf = find_rclone_conf(cfg) is not None
    for section in ("sync", "upload"):
        part = cfg.get(section)
        if isinstance(part, dict) and str(part.get("enabled", "")).lower() == "auto":
            part["enabled"] = has_conf
    return cfg


def base_config(name: str) -> Dict[str, Any]:
    """Cấu hình mặc định trong code + phần GHI ĐÈ ở thư mục dữ liệu (chỉ những dòng bạn đổi).

    Không dùng bản sao đầy đủ: bản sao sẽ giữ mãi mặc định cũ, cập nhật code không có tác dụng.
    """
    if os.path.isabs(name):
        return load_config(name)
    cfg = load_config(os.path.join(ROOT, name))
    custom = os.path.join(BASE, name)
    if os.path.exists(custom) and os.path.abspath(custom) != os.path.abspath(os.path.join(ROOT, name)):
        cfg = deep_merge(cfg, load_config(custom) or {})
    return cfg


def load_studio_config(name: str, project: Optional[str]) -> Dict[str, Any]:
    cfg = base_config(name)
    if project:
        project_cfg = os.path.join(DATA, "config.yaml")
        if os.path.exists(project_cfg):
            cfg = deep_merge(cfg, load_config(project_cfg) or {})
        cfg = fill_project(cfg, project)
        cfg = resolve_auto(cfg)
    else:
        for section in ("sync", "upload"):   # auto + {project} mà không có --project -> tắt Drive
            part = cfg.get(section, {})
            if str(part.get("enabled", "")).lower() == "auto" and "{project}" in json.dumps(part, ensure_ascii=False):
                part["enabled"] = False
        cfg = resolve_auto(cfg)
        for section in ("sync", "upload"):
            part = cfg.get(section, {})
            if part.get("enabled") and "{project}" in json.dumps(part, ensure_ascii=False):
                raise SystemExit(f"config.yaml mục '{section}' dùng {{project}}: hãy chạy với --project <tên bộ truyện>")
    return cfg


def config_path(name: str) -> str:
    """config.yaml trong thư mục dữ liệu (bản bạn đã sửa) được ưu tiên hơn bản mặc định của tool."""
    if os.path.isabs(name):
        return name
    custom = os.path.join(BASE, name)
    return custom if os.path.exists(custom) else os.path.join(ROOT, name)


def parse_range(text: Optional[str]) -> Tuple[int, int]:
    if not text:
        return 1, 10 ** 9
    a, _, b = text.partition("-")
    return int(a), int(b or a)


class Studio:
    def __init__(self, cfg: Dict[str, Any], args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        paths = cfg.get("paths", {})
        # Chương, audio, video, cache luôn nằm trong thư mục riêng của bộ: không bao giờ lẫn giữa các bộ.
        self.input_dir = own(paths.get("input_dir", "input"))
        self.audio_dir = own(paths.get("audio_dir", "output/audio"))
        self.video_dir = own(paths.get("video_dir", "output/video"))
        self.work_dir = own(paths.get("work_dir", "work"))
        for d in (self.audio_dir, self.video_dir, self.work_dir):
            os.makedirs(d, exist_ok=True)
        self.first, self.last = parse_range(args.range)
        self.master_cfg = cfg.get("master", {})
        self._audio: Optional[ChapterBuilder] = None
        self._render_queue: "queue.Queue[Optional[int]]" = queue.Queue()
        self._queued: set = set()
        self._lock = threading.Lock()
        self._last_sync = 0.0
        self.job = getattr(args, "job", None) or f"job{os.getpid()}"
        self.lock_dir = os.path.join(self.work_dir, "locks")
        self.jobs_dir = os.path.join(self.work_dir, "jobs")

    # ------------------------------------------------------------ jobs
    def heartbeat(self) -> str:
        path = os.path.join(self.jobs_dir, f"{self.job}.alive")
        os.makedirs(self.jobs_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": locks.HOST, "time": time.time()}, handle)
        return path

    def audio_jobs_alive(self) -> List[str]:
        """Các job đọc audio còn chạy (dựa vào file .alive và tiến trình còn sống)."""
        alive = []
        for path in glob.glob(os.path.join(self.jobs_dir, "*.alive")):
            if not locks.is_stale(path, stale_sec=3 * 3600):
                alive.append(os.path.basename(path)[:-6])
        return alive

    # ------------------------------------------------------------ Google Drive
    def rclone_config(self) -> Optional[str]:
        return find_rclone_conf(self.cfg) or rel(self.cfg.get("upload", {}).get("config_path") or "rclone.conf")

    def sync_from_drive(self, force: bool = False) -> None:
        """Tải chương mới + tài nguyên từ Drive, đánh dấu các tập đã có trên Drive để bỏ qua."""
        scfg = self.cfg.get("sync", {})
        if not scfg.get("enabled", False):
            return
        interval = float(scfg.get("refresh_minutes", 20)) * 60
        if not force and time.time() - self._last_sync < interval:
            return
        self._last_sync = time.time()
        conf = self.rclone_config()
        if scfg.get("input"):
            before = len(self.chapter_files())
            if drive.pull(scfg["input"], self.input_dir, conf, ["*.txt", "*.json"]):
                after = len(self.chapter_files())
                extra = f" (+{after - before} mới)" if after > before else ""
                log(f"Drive: {after} chương trong input{extra}", "Drive")
        for local, remote in (scfg.get("assets") or {}).items():
            if remote and drive.pull(remote, own(local), conf):   # giọng / video nền riêng từng bộ
                log(f"Drive: đã đồng bộ {local}", "Drive")
        remote_dir = self.cfg.get("upload", {}).get("remote_dir")
        if scfg.get("skip_uploaded", True) and remote_dir:
            names = drive.list_files(remote_dir, conf)
            if names is not None:
                marked = 0
                for name in drive.uploaded_episodes(names):
                    marker = os.path.join(self.video_dir, name + ".done")
                    if not os.path.exists(marker):
                        open(marker, "a").close()
                        marked += 1
                if marked:
                    log(f"Drive: {marked} tập đã có trên Drive, bỏ qua các chương của chúng.", "Drive")

    def episode_of(self, number: int) -> int:
        per = int(self.cfg.get("episode", {}).get("chapters_per_episode", 20))
        return (number - 1) // max(per, 1) + 1

    # ------------------------------------------------------------ chapters
    def chapter_files(self) -> List[Tuple[int, str]]:
        files = glob.glob(os.path.join(self.input_dir, "**", "*.txt"), recursive=True)
        items = []
        for path in files:
            if os.path.basename(path).startswith(("clone_voice", "_")):
                continue
            number = chapter_number(path)
            if number != 999999 and self.first <= number <= self.last:
                items.append((number, path))
        return sorted(items)

    def background_video(self) -> str:
        """Video nền theo config; không có thì lấy video đầu tiên trong cùng thư mục (tên tùy ý)."""
        path = rel(self.cfg.get("video", {}).get("background", "assets/background/background.mp4"))
        if os.path.exists(path):
            return path
        folder = os.path.dirname(path)
        videos = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                        if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")))
        if videos:
            log(f"Không thấy {os.path.basename(path)}, dùng video nền {os.path.basename(videos[0])}", "Video")
            return videos[0]
        return path

    def chapter_wav(self, number: int) -> str:
        return os.path.join(self.audio_dir, f"Chuong_{number:04d}.wav")

    # ------------------------------------------------------------ voice
    def voice_source(self) -> Optional[str]:
        vcfg = self.cfg.get("voice", {})
        for candidate in [vcfg.get("ref_audio", "")] + list(vcfg.get("fallbacks", [])):
            if candidate and os.path.exists(rel(candidate)):
                return rel(candidate)
        return None

    def audition_texts(self) -> List[str]:
        """Vài chunk thật của chương đầu để thử giọng (chunk đủ dài mới đo được độ giống giọng)."""
        files = self.chapter_files()
        if files:
            with open(files[0][1], "r", encoding="utf-8") as handle:
                text = chapter_text(handle.read(), self.load_extras()[0])
        else:
            text = self.BENCH_TEXT
        chunks, _ = chunk_text(text, int(self.cfg.get("tts", {}).get("max_chars", 256)))
        count = int(self.cfg.get("voice", {}).get("audition_sentences", 6))
        return [c for c in chunks if len(c) >= 40][:count]

    def voice(self) -> Dict[str, Any]:
        """Giọng mẫu -> đoạn câu trọn vẹn <= 7,8 s clone tốt nhất (chọn 1 lần, lưu trong work/voice).

        tts.voice đặt sẵn (model đã fine-tune giọng riêng) -> dùng thẳng giọng đó, không cần giọng mẫu.
        """
        tts_cfg = self.cfg.get("tts", {})
        if tts_cfg.get("voice"):
            engine = self.audio().tts
            engine.use_preset(str(tts_cfg["voice"]))
            log(f"Giọng: '{engine.voice_name}' của model {tts_cfg.get('backbone_repo') or 'VieNeu gốc'}", "Studio")
            return {"ref_audio": None, "fp": fingerprint(tts_cfg.get("backbone_repo"), engine.voice_name)}
        ref = self.voice_source()
        if not ref:
            raise RuntimeError("Không có giọng mẫu (voice.ref_audio): VieNeu cần một clip giọng để clone.")
        vcfg = self.cfg.get("voice", {})
        fp = fingerprint(file_fingerprint(ref), VOICE_PICK_VERSION, bool(tts_cfg.get("denoise", True)),
                         bool(vcfg.get("audition", True)))
        picked = os.path.join(self.work_dir, "voice", f"ref_{fp}.wav")
        meta_path = os.path.splitext(picked)[0] + ".json"
        engine = self.audio().tts
        if not os.path.exists(picked):
            os.makedirs(os.path.dirname(picked), exist_ok=True)
            # Nhiều job cùng lúc: chỉ một job chọn giọng mẫu, job khác chờ kết quả.
            lock = picked + ".lock"
            if locks.claim(lock, self.job, stale_sec=1800):
                try:
                    texts = self.audition_texts() if vcfg.get("audition", True) else []
                    meta = engine.pick_reference(ref, picked, texts)
                    write_json(meta_path, {"source": os.path.basename(ref), **meta})
                finally:
                    locks.release(lock)
            else:
                log("Job khác đang chọn giọng mẫu, chờ...", "Giọng")
                for _ in range(360):
                    if os.path.exists(picked):
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError("Hết thời gian chờ job khác chọn giọng mẫu.")
        engine.set_voice(picked)
        meta = read_json(meta_path, {}) or {}
        sim = meta.get("similarity")
        log(f"Giọng mẫu: {os.path.basename(ref)} -> đoạn {meta.get('segment')} ({meta.get('seconds')}s"
            f"{f', giống giọng gốc {sim:.3f}' if sim is not None else ''})", "Studio")
        return {"ref_audio": picked, "fp": fp}

    def progress(self) -> Dict[str, Any]:
        """Tiến độ thật của bộ truyện: chương nào chưa có audio mà tập của nó cũng chưa lên Drive."""
        files = self.chapter_files()
        last = max((n for n, _ in files), default=0)
        episodes = sorted({self.episode_of(n) for n, _ in files})
        done = [e for e in episodes if self.episode_done(e, last)]
        pending = [n for n, _ in files
                   if not os.path.exists(self.chapter_wav(n)) and not self.episode_done(self.episode_of(n), last)]
        return {"chapters": len(files), "episodes": len(episodes), "episodes_done": len(done),
                "chapters_pending": len(pending), "first_pending": pending[0] if pending else None}

    def audio(self) -> ChapterBuilder:
        if self._audio is None:
            tts_cfg = dict(self.cfg.get("tts", {}))
            tts = VieneuEngine(tts_cfg)
            asr = ASRChecker(self.cfg.get("asr", {}))
            self._audio = ChapterBuilder(tts, asr, tts_cfg, self.cfg.get("asr", {}), self.master_cfg)
        return self._audio

    def load_extras(self) -> Tuple[Dict[str, str], List[str]]:
        glossary: Dict[str, str] = {}
        path = os.path.join(self.input_dir, "glossary.json")
        if os.path.exists(path):
            glossary = read_json(path, {}) or {}
        names: List[str] = []
        names_path = os.path.join(self.input_dir, "names.txt")
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
        return glossary, names

    def process_chapter(self, number: int, path: str, voice: Dict[str, Any], glossary, names) -> Optional[float]:
        out = self.chapter_wav(number)
        if os.path.exists(out) and not self.args.force:
            return None
        started = time.time()
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        text = chapter_text(raw, glossary)
        if not text:
            log(f"Chương {number}: rỗng sau khi làm sạch, bỏ qua.", "Cảnh báo")
            return None
        builder = self.audio()
        builder.force = bool(self.args.force)
        builder.ignore_terms = names + [item["name"] for item in extract_standardization_table(raw)]
        log(f"===== Chương {number}: {len(text)} ký tự =====", "Studio")
        timeline = builder.build(text, voice["fp"], out,
                                 os.path.join(self.work_dir, f"ch{number:04d}", "timeline.json"))
        spent = time.time() - started
        log(f"Chương {number}: {format_duration(timeline['total_duration_sec'])} audio trong "
            f"{format_duration(spent)} (x{timeline['total_duration_sec'] / max(spent, 1):.1f} thời gian thực)", "Studio")
        return timeline["total_duration_sec"]

    def pending_chapters(self, handled: set) -> List[Tuple[int, str]]:
        """Chương còn phải đọc: chưa có audio, chưa xử lý trong lần chạy này, tập chưa lên Drive."""
        files = self.chapter_files()
        last = max((n for n, _ in files), default=0)
        return [
            (n, p) for n, p in files
            if n not in handled
            and (self.args.force or not os.path.exists(self.chapter_wav(n)))
            and not self.episode_done(self.episode_of(n), last)
        ]

    def run_audio(self, with_video: bool) -> None:
        alive = self.heartbeat()
        try:
            self._run_audio(with_video, alive)
        finally:
            if os.path.exists(alive):
                os.remove(alive)

    def _run_audio(self, with_video: bool, alive: str) -> None:
        self.sync_from_drive(force=True)
        chapters = self.chapter_files()
        if not chapters:
            log(f"Không có file .txt nào trong {self.input_dir}.", "Studio")
            return
        handled: set = set()
        pending = self.pending_chapters(handled)
        log(f"[{self.job}] {len(chapters)} chương, cần đọc {len(pending)} (chia với các job khác).", "Studio")
        if with_video:
            self.queue_ready_episodes()
        voice = glossary = names = None
        while pending:
            for number, path in pending:
                handled.add(number)
                lock = os.path.join(self.lock_dir, f"ch{number:04d}.lock")
                # Chương đang được job khác đọc -> bỏ qua. Kiểm tra lại audio sau khi giành khóa
                # vì job khác có thể vừa đọc xong.
                if not locks.claim(lock, self.job):
                    continue
                try:
                    if not self.args.force and os.path.exists(self.chapter_wav(number)):
                        continue
                    if voice is None:
                        voice = self.voice()
                        glossary, names = self.load_extras()
                    self.process_chapter(number, path, voice, glossary, names)
                except Exception as exc:
                    log(f"Chương {number} lỗi: {exc}", "Lỗi")
                    if self.args.stop_on_error:
                        raise
                    continue
                finally:
                    locks.release(lock)
                    locks.touch(alive)
                if with_video:
                    self.queue_ready_episodes()
                self.sync_from_drive()   # theo chu kỳ: lấy chương mới vừa thêm lên Drive
            self.sync_from_drive(force=True)
            pending = self.pending_chapters(handled)
            if pending:
                log(f"Có thêm {len(pending)} chương mới từ Drive, đọc tiếp.", "Studio")
        builder = self._audio
        if builder:
            builder.tts.unload()
            builder.asr.unload()

    # ------------------------------------------------------------ episodes
    def episode_info(self, episode: int) -> Tuple[List[int], str]:
        per = int(self.cfg.get("episode", {}).get("chapters_per_episode", 20))
        numbers = list(range((episode - 1) * per + 1, episode * per + 1))
        return numbers, os.path.join(self.video_dir, f"Chuong_{numbers[0]:04d}-{numbers[-1]:04d}_Tap{episode:03d}.mp4")

    def episode_done(self, episode: int, last_chapter: Optional[int] = None) -> bool:
        """Đã có video (hoặc dấu .done sau khi upload) phủ ĐỦ các chương của tập này.

        Tập lẻ cuối (ít chương hơn) chỉ tính là xong khi nó phủ tới chương cuối hiện có.
        Nhờ vậy đổi số chương/tập (10 -> 20) thì tập cũ 0001-0010 không che mất chương 11-20.
        """
        numbers, _ = self.episode_info(episode)
        pattern = os.path.join(self.video_dir, f"Chuong_{numbers[0]:04d}-*_Tap{episode:03d}.mp4*")
        for p in glob.glob(pattern):
            if not p.endswith((".mp4", ".done")):
                continue
            m = re.search(r"Chuong_\d+-(\d+)_Tap", os.path.basename(p))
            end = int(m.group(1)) if m else 0
            if end >= numbers[-1]:
                return True
            if last_chapter is None:
                last_chapter = max((n for n, _ in self.chapter_files()), default=0)
            if end >= min(last_chapter, numbers[-1]):
                return True
        return False

    def ready_episodes(self, finish: bool) -> List[int]:
        per = int(self.cfg.get("episode", {}).get("chapters_per_episode", 20))
        chapters = [n for n, _ in self.chapter_files()]
        if not chapters or per <= 0:
            return []
        last_chapter = max(chapters)
        ready = []
        for episode in sorted({(n - 1) // per + 1 for n in chapters}):
            numbers, _ = self.episode_info(episode)
            if self.episode_done(episode):
                continue
            have = [n for n in numbers if os.path.exists(self.chapter_wav(n))]
            if len(have) == per:
                ready.append(episode)
            elif finish and have and numbers[0] <= last_chapter <= numbers[-1] and \
                    all(os.path.exists(self.chapter_wav(n)) for n in numbers if n <= last_chapter):
                ready.append(episode)
        return ready

    def make_episode(self, episode: int) -> None:
        """Ghép + upload một tập; chỉ một job được làm một tập (khóa tapNNN.lock)."""
        lock = os.path.join(self.lock_dir, f"tap{episode:03d}.lock")
        if not locks.claim(lock, self.job):
            return
        try:
            if not self.episode_done(episode):
                self._make_episode(episode)
        finally:
            locks.release(lock)

    def _make_episode(self, episode: int) -> None:
        numbers, _ = self.episode_info(episode)
        present = [n for n in numbers if os.path.exists(self.chapter_wav(n))]
        wavs = [self.chapter_wav(n) for n in present]
        out = os.path.join(self.video_dir, f"Chuong_{present[0]:04d}-{present[-1]:04d}_Tap{episode:03d}.mp4")
        ecfg = self.cfg.get("episode", {})
        inserts = [dict(item, file=rel(item["file"])) for item in ecfg.get("inserts", []) if item.get("file")]
        maker = VideoMaker(self.cfg.get("video", {}), os.path.join(self.work_dir, "video"), self.master_cfg)
        started = time.time()
        total = maker.make_episode(wavs, self.background_video(),
                                   out, inserts)
        log(f"XONG tập {episode}: {os.path.basename(out)} ({format_duration(total)}, ghép trong "
            f"{format_duration(time.time() - started)})", "Video")
        ucfg = dict(self.cfg.get("upload", {}))
        ucfg["config_path"] = self.rclone_config()
        if drive.upload(out, ucfg, os.path.join(self.work_dir, "upload.log")):
            open(out + ".done", "a").close()
            if ucfg.get("delete_after_upload", True):
                os.remove(out)
            if ucfg.get("delete_audio_after_upload", False):
                for wav in wavs:
                    os.remove(wav)

    def queue_ready_episodes(self, finish: bool = False) -> None:
        for episode in self.ready_episodes(finish or self.args.finish):
            with self._lock:
                if episode in self._queued:
                    continue
                self._queued.add(episode)
            self._render_queue.put(episode)

    def _render_worker(self) -> None:
        while True:
            episode = self._render_queue.get()
            if episode is None:
                return
            try:
                self.make_episode(episode)
            except Exception as exc:
                log(f"Tập {episode} lỗi: {exc}", "Lỗi")

    def retry_uploads(self) -> None:
        """Video tập đã ghép nhưng upload lỗi (mất mạng...) -> thử đẩy lại."""
        ucfg = dict(self.cfg.get("upload", {}))
        if not ucfg.get("enabled", False):
            return
        ucfg["config_path"] = self.rclone_config()
        for out in sorted(glob.glob(os.path.join(self.video_dir, "Chuong_*_Tap*.mp4"))):
            if os.path.exists(out + ".done"):
                continue
            match = drive.EPISODE_RE.match(os.path.basename(out))
            if not match:
                continue
            lock = os.path.join(self.lock_dir, f"tap{int(match.group(3)):03d}.lock")
            if not locks.claim(lock, self.job):
                continue
            try:
                if drive.upload(out, ucfg, os.path.join(self.work_dir, "upload.log")):
                    open(out + ".done", "a").close()
                    if ucfg.get("delete_after_upload", True):
                        os.remove(out)
            finally:
                locks.release(lock)

    def watch_video(self) -> None:
        """Job ghép video chạy ngầm: đủ chương là ghép + upload; dừng khi các job audio xong hết."""
        poll = float(self.cfg.get("episode", {}).get("watch_seconds", 60))
        log("Job video: theo dõi audio, đủ chương là ghép tập.", "Video")
        idle_rounds = 0
        failures: Dict[int, int] = {}
        while True:
            self.sync_from_drive()
            self.retry_uploads()
            ready = [e for e in self.ready_episodes(finish=False) if failures.get(e, 0) < 3]
            progressed = False
            for episode in ready:
                try:
                    self.make_episode(episode)
                except Exception as exc:
                    failures[episode] = failures.get(episode, 0) + 1
                    log(f"Tập {episode} lỗi (lần {failures[episode]}/3): {exc}", "Lỗi")
                progressed |= self.episode_done(episode)
            if progressed:
                idle_rounds = 0
                continue   # vừa xong tập: xem ngay còn tập nào đủ chương không
            if not self.audio_jobs_alive():
                idle_rounds += 1
                # Chờ thêm 1 vòng để chắc các job audio đã thật sự dừng (không phải đang khởi động).
                if idle_rounds >= 2:
                    if self.args.finish:
                        for episode in self.ready_episodes(finish=True):
                            self.make_episode(episode)
                    missing = [e for e in {self.episode_of(n) for n, _ in self.chapter_files()}
                               if not self.episode_done(e)]
                    if missing:
                        log(f"Job video dừng. Tập chưa đủ audio: {sorted(missing)[:10]} "
                            "(tập lẻ cuối: chạy video --finish).", "Video")
                    else:
                        log("Job video dừng: mọi tập đã xong.", "Video")
                    return
            time.sleep(poll)

    def run(self, audio: bool, video: bool) -> None:
        if not audio and getattr(self.args, "watch", False):
            self.sync_from_drive(force=True)
            self.watch_video()
            return
        if not audio:
            self.sync_from_drive(force=True)
        worker = None
        if video:
            # Ghép video dùng CPU nên chạy song song trong lúc GPU tiếp tục đọc audio.
            worker = threading.Thread(target=self._render_worker, daemon=True)
            worker.start()
        if audio:
            self.run_audio(with_video=video)
        if video:
            self.retry_uploads()   # video đã ghép nhưng lần trước upload lỗi -> đẩy lại
            self.queue_ready_episodes(finish=self.args.finish)
            self._render_queue.put(None)
            if self._render_queue.qsize() > 1:
                log("Đang chờ ghép xong các tập còn lại...", "Video")
            worker.join()

    # ------------------------------------------------------------ bench
    BENCH_TEXT = (
        "Dưới ánh nắng vàng nhạt của buổi chiều tà, gió từ vùng ngoại ô thổi về mang theo chút hanh khô "
        "đặc trưng của phương Bắc. Trên bậc thềm đá rộng lớn trước quảng trường, bóng của hắn kéo dài đổ "
        "xuống, cô độc nhưng mang một áp lực vô hình khiến người đối diện không dám nhìn thẳng.\n"
        "“Năm 2003, giá đất ở đây chỉ khoảng 3.500 tệ một mét vuông, ngươi tin không?”\n"
        "Lâm Viễn ngẩn người, rất lâu sau mới khẽ gật đầu. Hắn biết, người đứng trước mặt mình đã nhìn thấy "
        "một bàn cờ mà cả thành phố này còn chưa ai nhận ra.\n"
    )

    def bench(self) -> None:
        """Đo tốc độ đọc + đánh giá bước nghe lại (ASR) trên chương thật.

        Mỗi chương: VieNeu đọc một lượt (= tốc độ khi TẮT ASR), rồi ASR nghe lại toàn bộ chunk
        (= chi phí khi BẬT ASR) và đọc lại chunk nghi lỗi. Chunk bị gắn cờ được lưu cả bản lượt
        đầu lẫn bản cuối vào work/bench/<giờ>/ để nghe xem lỗi thật hay ASR nghe nhầm.
        """
        files = self.chapter_files()[: max(1, int(self.args.chapters))]
        glossary, names = self.load_extras()
        items = []
        for number, path in files:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            items.append((f"Chuong_{number:04d}", chapter_text(raw, glossary),
                          [i["name"] for i in extract_standardization_table(raw)]))
        if not items:
            items = [("van_ban_mau", self.BENCH_TEXT * 6, [])]
        builder = self.audio()
        if self.args.no_asr:
            builder.asr.enabled = False
        started = time.time()
        voice = self.voice()
        voice_sec = time.time() - started
        # Khởi động (CUDA graph, nạp ASR) không tính vào thời gian đo.
        warm, _ = builder.tts.synthesize_chunks(["Xin chào các bạn, đây là câu khởi động để đo tốc độ."])
        if builder.asr.enabled:
            builder.asr.transcribe_batch([(warm[0], builder.tts.sample_rate)])
        out_dir = os.path.join(self.work_dir, "bench", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        sr = builder.tts.sample_rate
        max_cer = float(self.cfg.get("asr", {}).get("max_cer", 0.12))
        report: Dict[str, Any] = {"backend": builder.tts.backend, "voice": voice,
                                  "voice_pick_sec": round(voice_sec, 1), "asr": builder.asr.enabled,
                                  "asr_model": builder.asr.model_id, "max_cer": max_cer, "chapters": []}
        totals = {"audio": 0.0, "tts1": 0.0, "tts": 0.0, "asr": 0.0, "chunks": 0, "flag1": 0, "flag": 0}
        cers: List[float] = []
        for name, text, extra_names in items:
            builder.ignore_terms = names + extra_names
            chunks, gaps = chunk_text(text, int(self.cfg.get("tts", {}).get("max_chars", 256)))
            res = builder.render(chunks)
            stats = res["stats"]
            final = join_chunks(res["wavs"], gaps)
            write_wav(os.path.join(out_dir, f"{name}.wav"), final, sr)
            audio_sec = final.size / sr
            flagged = []
            for i, (c1, c2) in enumerate(zip(res["first_checks"], res["checks"])):
                if c1.get("cer") is not None:
                    cers.append(c1["cer"])
                if not c1.get("ok", True):
                    base = os.path.join(out_dir, f"{name}_chunk{i:03d}")
                    write_wav(base + "_luot1.wav", res["first_wavs"][i], sr)
                    write_wav(base + "_cuoi.wav", res["wavs"][i], sr)
                    flagged.append({"chunk": i, "text": chunks[i], "luot1": c1, "cuoi": c2})
            still = sum(1 for c in res["checks"] if not c.get("ok", True))
            report["chapters"].append({
                "name": name, "chars": len(text), "chunks": len(chunks), "audio_sec": round(audio_sec, 1),
                "tts_first_sec": round(stats["tts_first_sec"], 1), "tts_total_sec": round(stats["tts_sec"], 1),
                "asr_sec": round(stats["asr_sec"], 1), "flagged_first": len(flagged), "flagged_final": still,
                "flagged": flagged})
            totals["audio"] += audio_sec
            totals["tts1"] += stats["tts_first_sec"]
            totals["tts"] += stats["tts_sec"]
            totals["asr"] += stats["asr_sec"]
            totals["chunks"] += len(chunks)
            totals["flag1"] += len(flagged)
            totals["flag"] += still
            log(f"{name}: {audio_sec / 60:.1f} phút audio, {len(chunks)} chunk | đọc {stats['tts_first_sec']:.0f}s"
                + (f", ASR {stats['asr_sec']:.0f}s, đọc lại {stats['tts_sec'] - stats['tts_first_sec']:.0f}s | "
                   f"nghi lỗi {len(flagged)} -> còn {still}" if builder.asr.enabled else ""), "Bench")
        no_asr = totals["tts1"]
        with_asr = totals["tts"] + totals["asr"]
        report["machine"] = self._machine()
        summary = {
            "audio_min": round(totals["audio"] / 60, 2),
            "chunks": totals["chunks"],
            "speed_no_asr_x": round(totals["audio"] / max(no_asr, 1e-6), 1),
            "speed_with_asr_x": round(totals["audio"] / max(with_asr, 1e-6), 1) if builder.asr.enabled else None,
            "asr_overhead_pct": round(100 * (with_asr - no_asr) / max(no_asr, 1e-6)) if builder.asr.enabled else None,
            "flagged_first_pct": round(100 * totals["flag1"] / max(totals["chunks"], 1), 2),
            "flagged_final_pct": round(100 * totals["flag"] / max(totals["chunks"], 1), 2),
        }
        if cers:
            summary.update({f"cer_{k}": round(float(v), 4) for k, v in (
                ("mean", np.mean(cers)), ("p50", np.percentile(cers, 50)),
                ("p90", np.percentile(cers, 90)), ("p99", np.percentile(cers, 99)))})
        report["summary"] = summary
        write_json(os.path.join(out_dir, "report.json"), report)
        per = int(self.cfg.get("episode", {}).get("chapters_per_episode", 20))
        chapter_audio_min = totals["audio"] / 60 / max(len(items), 1)
        lines = [f"Máy: {report['machine']} | VieNeu backend {builder.tts.backend} | chọn giọng mẫu {voice_sec:.0f}s",
                 f"{len(items)} chương = {summary['audio_min']:.1f} phút audio, {totals['chunks']} chunk"]
        modes = [("TẮT ASR", no_asr)] + ([("BẬT ASR", with_asr)] if builder.asr.enabled else [])
        for label, secs in modes:
            speed = totals["audio"] / max(secs, 1e-6)
            chapter_min = chapter_audio_min / speed
            line = (f"{label}: nhanh gấp {speed:.1f} lần thời gian thực -> 1 chương ~{chapter_min * 60:.0f}s, "
                    f"1 tập {per} chương ~{chapter_min * per:.1f} phút")
            if self.args.price:
                line += f", ~${float(self.args.price) * chapter_min * per / 60:.3f}/tập"
            lines.append(line)
        if builder.asr.enabled and cers:
            lines += [f"ASR tốn thêm {summary['asr_overhead_pct']}% thời gian so với chỉ đọc",
                      f"Sai ký tự (CER) lượt đầu: trung bình {summary['cer_mean']:.1%}, trung vị "
                      f"{summary['cer_p50']:.1%}, p90 {summary['cer_p90']:.1%}, p99 {summary['cer_p99']:.1%}",
                      f"Chunk nghi lỗi (CER > {max_cer:.0%} hoặc lặp từ): lượt đầu {totals['flag1']} "
                      f"({summary['flagged_first_pct']}%), sau đọc lại còn {totals['flag']} "
                      f"({summary['flagged_final_pct']}%)"]
        lines.append(f"Chi tiết + file nghe thử: {out_dir}")
        for line in lines:
            log(line, "Bench")

    @staticmethod
    def _machine() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                return f"{props.name} {props.total_memory / 1024 ** 3:.0f} GB (VRAM dùng tối đa {peak:.1f} GB)"
        except ImportError:
            pass
        return f"CPU {os.cpu_count()} luồng"

    # ------------------------------------------------------------ check
    def check(self) -> None:
        import shutil

        ok = True
        for tool in ("ffmpeg", "ffprobe"):
            found = shutil.which(tool)
            log(f"{tool}: {found or 'THIẾU'}", "Check")
            ok &= bool(found)
        log(f"rclone: {shutil.which('rclone') or 'không có (không upload được)'}", "Check")
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                log(f"GPU: {props.name} {props.total_memory / 1024 ** 3:.0f} GB, bf16={torch.cuda.is_bf16_supported()}", "Check")
            else:
                log("GPU: không có (VieNeu chạy CPU/ONNX, chậm hơn GPU ~25 lần)", "Check")
        except ImportError:
            log("torch: chưa cài (VieNeu chạy CPU/ONNX)", "Check")
        try:
            from importlib.metadata import version

            log(f"vieneu: {version('vieneu')}", "Check")
        except Exception:  # noqa: BLE001
            log("vieneu: THIẾU (pip install -r requirements.txt)", "Check")
            ok = False
        vcfg = self.cfg.get("voice", {})
        log(f"Giọng mẫu: {rel(vcfg.get('ref_audio', '')) if os.path.exists(rel(vcfg.get('ref_audio', ''))) else 'THIẾU'}", "Check")
        bg = self.background_video()
        log(f"Video nền: {bg if os.path.exists(bg) else 'THIẾU'}", "Check")
        log(f"Chương trong {self.input_dir}: {len(self.chapter_files())}", "Check")
        log("Môi trường OK." if ok else "Thiếu công cụ bắt buộc.", "Check")


def resolve_project(config_name: str, project: str) -> str:
    """Gõ tên không dấu (Dai Niet Ban) -> đúng tên thư mục trên Drive (Đại Niết Bàn).

    Web Terminal của RunPod hay làm rơi chữ có dấu, nên tool tự tìm thư mục khớp khi bỏ dấu.
    """
    project = project.strip()
    try:
        cfg = resolve_auto(base_config(config_name))
        scfg = cfg.get("sync", {})
        template = str(scfg.get("input", ""))
        if not scfg.get("enabled") or "{project}" not in template:
            return project
        parent = template.split("{project}")[0].rstrip("/")
        names = drive.list_dirs(parent, find_rclone_conf(cfg))
    except Exception as exc:  # noqa: BLE001 - không đọc được Drive thì giữ nguyên tên
        log(f"Không dò được tên bộ trên Drive: {exc}", "Drive")
        return project
    if not names:
        return project
    found = drive.match_folder(project, names)
    if found is None:
        log(f"KHÔNG thấy thư mục '{project}' trong {parent}. Các bộ đang có: {', '.join(names)}", "Lỗi")
        raise SystemExit(1)
    if found != project:
        log(f"Tên bộ '{project}' -> thư mục Drive '{found}'", "Drive")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "audio", "video", "check", "bench", "resolve", "progress"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--range", help="chỉ xử lý chương a-b, ví dụ 1-400")
    parser.add_argument("--finish", action="store_true", help="gom cả tập lẻ cuối truyện")
    parser.add_argument("--force", action="store_true", help="đọc lại cả chương đã có audio")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--price", type=float, help="bench: giá pod $/giờ để tính chi phí")
    parser.add_argument("--chapters", type=int, default=2, help="bench: số chương thật để đo (mặc định 2)")
    parser.add_argument("--no-asr", action="store_true", help="bench: chỉ đo tốc độ đọc, không nghe lại")
    parser.add_argument("--project", help="tên bộ truyện: dữ liệu riêng ở projects/<tên>, {project} trong config")
    parser.add_argument("--job", help="tên job (log, khóa), ví dụ A, B")
    parser.add_argument("--watch", action="store_true", help="video: chạy ngầm, đủ chương là ghép tập")
    args = parser.parse_args()
    global DATA
    if args.project:
        args.project = resolve_project(args.config, args.project)
        if args.command == "resolve":      # run.sh hỏi tên thư mục thật của bộ truyện
            print(args.project)
            return 0
        DATA = os.path.join(BASE, "projects", args.project)
        os.makedirs(DATA, exist_ok=True)
    cfg = load_studio_config(args.config, args.project)
    studio = Studio(cfg, args)
    tag = f"_{args.job}" if args.job else ""
    set_log_file(os.path.join(studio.work_dir, "logs", time.strftime(f"studio{tag}_%Y%m%d.log")))
    if args.project:
        log(f"Bộ truyện: {args.project} ({DATA})", "Studio")
    if args.command == "check":
        studio.check()
    elif args.command == "bench":
        studio.bench()
    elif args.command == "progress":   # script tắt pod hỏi: còn chương nào chưa xong không?
        print(json.dumps(studio.progress(), ensure_ascii=False))
    elif args.command == "audio":
        studio.run(audio=True, video=False)
    elif args.command == "video":
        studio.run(audio=False, video=True)
    else:
        studio.run(audio=True, video=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
