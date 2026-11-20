"""Ghép tập: audio các chương (+ intro/watermark) lên video nền ngắn lặp đến hết audio.

Vì sao bản cũ bị sướt / mất tiếng:
  * Audio bị nén mất dữ liệu 3 lần (MP3 chương -> MP3 tập 128k -> AAC 128k).
  * Nối MP3 để lại khoảng hụt (padding của encoder) ở mỗi chỗ nối.
  * `-stream_loop -1 -c:v copy` làm timestamp gãy ở mỗi lần lặp -> khựng hình, có trình
    phát mất tiếng sau điểm lặp.
Cách làm ở đây:
  * Audio chương là WAV không nén; nối bằng concat demuxer (chính xác tới từng mẫu) và
    chỉ nén AAC MỘT lần ở bước cuối, 48 kHz stereo 192k.
  * Video nền được chuẩn hóa MỘT lần thành "đơn vị lặp" (fps cố định, keyframe đầu, không
    B-frame, tùy chọn crossfade cuối->đầu để lặp liền mạch), rồi lặp bằng danh sách concat
    copy -> timestamp liên tục, gần như không tốn CPU.
"""
import math
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from core.audio_master import master_wav
from core.utils import file_fingerprint, fingerprint, log, read_json, write_json


class VideoError(RuntimeError):
    pass


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise VideoError("Không tìm thấy ffmpeg.")
    return path


def _run(cmd: List[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise VideoError(f"FFmpeg lỗi ({res.returncode}): {res.stderr[-1500:]}")


def probe(path: str, entries: str = "format=duration", stream: Optional[str] = None) -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoError("Không tìm thấy ffprobe.")
    cmd = [ffprobe, "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "default=nw=1:nk=1", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()


def media_duration(path: str) -> float:
    try:
        return float(probe(path).splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def wav_duration(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return media_duration(path)


def _concat_line(path: str) -> str:
    safe = os.path.abspath(path).replace(os.sep, "/").replace("'", "'\\''")
    return f"file '{safe}'\n"


class VideoMaker:
    def __init__(self, cfg: Dict[str, Any], work_dir: str, master_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.work_dir = work_dir
        self.master_cfg = master_cfg or {}
        os.makedirs(work_dir, exist_ok=True)

    # ------------------------------------------------------------ video nền
    def _video_encode_args(self) -> List[str]:
        fps = int(self.cfg.get("fps", 30))
        return [
            "-c:v", "libx264", "-preset", str(self.cfg.get("x264_preset", "medium")),
            "-crf", str(int(self.cfg.get("crf", 18))), "-pix_fmt", "yuv420p",
            "-g", str(fps * 2), "-bf", "0", "-video_track_timescale", "90000",
        ]

    def loop_unit(self, background: str) -> Tuple[str, float]:
        """Chuẩn hóa video nền thành đơn vị lặp (làm một lần, có cache)."""
        if not os.path.exists(background):
            raise VideoError(f"Không có video nền: {background}")
        keys = ("fps", "crf", "x264_preset", "resolution", "loop_crossfade_sec")
        fp = fingerprint(file_fingerprint(background), {k: self.cfg.get(k) for k in keys}, 2)
        out = os.path.join(self.work_dir, f"loop_unit_{fp}.mp4")
        meta_path = f"{out}.json"
        meta = read_json(meta_path)
        if meta and os.path.exists(out):
            return out, float(meta["duration"])

        fps = int(self.cfg.get("fps", 30))
        src_dur = media_duration(background)
        if src_dur <= 0.5:
            raise VideoError(f"Video nền hỏng hoặc quá ngắn: {background}")
        vf = [f"fps={fps}"]
        res = self.cfg.get("resolution")
        if res:
            w, h = (int(v) for v in res)
            vf.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
        vf.append("format=yuv420p")
        chain = ",".join(vf)
        xf = float(self.cfg.get("loop_crossfade_sec", 1.0) or 0)
        xf = min(xf, src_dur / 4)
        log(f"Chuẩn hóa video nền thành đơn vị lặp ({src_dur:.1f}s, crossfade {xf:.1f}s)...", "Video")
        tmp = f"{out}.tmp.mp4"
        if xf >= 0.1:
            # Phần [xf, D] rồi crossfade đuôi sang [0, xf]: đơn vị dài D-xf, cuối khớp liền với đầu.
            graph = (
                f"[0:v]{chain},split[a][b];"
                f"[a]trim=start={xf:.3f},setpts=PTS-STARTPTS[main];"
                f"[b]trim=end={xf:.3f},setpts=PTS-STARTPTS[head];"
                f"[main][head]xfade=transition=fade:duration={xf:.3f}:offset={src_dur - 2 * xf:.3f}[v]"
            )
            cmd = [_ffmpeg(), "-y", "-i", background, "-filter_complex", graph, "-map", "[v]", "-an"]
        else:
            cmd = [_ffmpeg(), "-y", "-i", background, "-vf", chain, "-an"]
        _run(cmd + self._video_encode_args() + ["-movflags", "+faststart", tmp])
        os.replace(tmp, out)
        duration = media_duration(out)
        write_json(meta_path, {"duration": duration, "source": background})
        return out, duration

    # ------------------------------------------------------------ audio tập
    def _as_wav(self, path: str, sr: int) -> str:
        """Đổi file chèn (intro/watermark mp3...) sang WAV cùng định dạng, cùng độ to với chương."""
        fp = fingerprint(file_fingerprint(path), sr, self.master_cfg, 2)
        out = os.path.join(self.work_dir, f"insert_{fp}.wav")
        if os.path.exists(out):
            return out
        tmp = f"{out}.src.wav"
        _run([_ffmpeg(), "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s24le", tmp])
        if not master_wav(tmp, out, self.master_cfg):
            os.replace(tmp, out)
        elif os.path.exists(tmp):
            os.remove(tmp)
        return out

    def _silence(self, seconds: float, sr: int) -> str:
        out = os.path.join(self.work_dir, f"silence_{sr}_{int(seconds * 1000)}.wav")
        if not os.path.exists(out):
            _run([_ffmpeg(), "-y", "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono", "-t", f"{seconds:.3f}",
                  "-c:a", "pcm_s24le", out])
        return out

    def episode_audio_parts(self, chapter_wavs: List[str], inserts: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Thứ tự các file audio của tập: chèn intro/watermark theo `after` (sau chương thứ N)."""
        import soundfile as sf

        sr = sf.info(chapter_wavs[0]).samplerate
        count = len(chapter_wavs)
        before: Dict[int, List[str]] = {}   # chèn TRƯỚC chương thứ index (0 = đầu tập)
        ending: List[str] = []              # after: "end" -> cuối tập
        for item in inserts or []:
            path = item.get("file")
            if not path:
                continue
            if not os.path.exists(path):
                log(f"Không thấy file chèn {path}, bỏ qua.", "Cảnh báo")
                continue
            after = item.get("after", 0)
            if str(after).lower() == "end":
                ending.append(self._as_wav(path, sr))
            elif int(after) < count:   # watermark "sau chương 4" chỉ chèn khi còn chương phía sau
                before.setdefault(int(after), []).append(self._as_wav(path, sr))
        gap = float(self.cfg.get("insert_gap_sec", 0.4))
        pad = [self._silence(gap, sr)] if gap > 0 else []
        parts: List[str] = []
        for index, wav in enumerate(chapter_wavs):
            for insert in before.get(index, []):
                parts += [insert] + pad
            parts.append(wav)
        for insert in ending:
            parts += pad + [insert]
        for wav in parts:
            if sf.info(wav).samplerate != sr or sf.info(wav).channels != 1:
                raise VideoError(f"Audio {wav} khác định dạng ({sf.info(wav).samplerate} Hz)")
        return parts

    # ------------------------------------------------------------ ghép
    def make_episode(
        self,
        chapter_wavs: List[str],
        background: str,
        out_mp4: str,
        inserts: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Tạo video tập. Trả về thời lượng (giây)."""
        if not chapter_wavs:
            raise VideoError("Không có audio chương nào.")
        parts = self.episode_audio_parts(chapter_wavs, inserts)
        total = sum(wav_duration(p) for p in parts)
        unit, unit_dur = self.loop_unit(background)
        repeats = int(math.ceil(total / max(unit_dur, 0.1))) + 1

        stem = os.path.splitext(os.path.abspath(out_mp4))[0]
        audio_list, video_list = f"{stem}.audio.txt", f"{stem}.video.txt"
        with open(audio_list, "w", encoding="utf-8") as handle:
            handle.writelines(_concat_line(p) for p in parts)
        with open(video_list, "w", encoding="utf-8") as handle:
            handle.writelines(_concat_line(unit) for _ in range(repeats))
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
        tmp = f"{stem}.partial.mp4"
        bitrate = str(self.cfg.get("audio_bitrate", "192k"))
        cmd = [
            _ffmpeg(), "-y", "-hide_banner",
            "-f", "concat", "-safe", "0", "-i", video_list,
            "-f", "concat", "-safe", "0", "-i", audio_list,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-af", "aresample=48000", "-ac", "2", "-c:a", "aac", "-b:a", bitrate,
            "-t", f"{total:.3f}", "-movflags", "+faststart", tmp,
        ]
        log(f"Ghép video {total / 60:.1f} phút ({len(chapter_wavs)} chương, lặp nền {repeats} lần)...", "Video")
        try:
            _run(cmd)
            got = media_duration(tmp)
            if abs(got - total) > 1.0:
                raise VideoError(f"Video ra {got:.1f}s nhưng audio {total:.1f}s")
            os.replace(tmp, out_mp4)
        finally:
            for path in (audio_list, video_list, tmp):
                if os.path.exists(path):
                    os.remove(path)
        return total
