import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from core.asr_checker import char_error_rate, repeated_words
from core.audio_master import measure_loudness
from core.audio_utils import read_wav, to_mono_float, write_wav
from core.chapter_builder import ChapterBuilder
from core.text_processor import apply_glossary, chapter_text, clean_paragraphs

SR = 24000
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
try:
    import vieneu_utils  # noqa: F401  (pyflakes: dùng để dò gói)
    HAS_VIENEU = True
except ImportError:
    HAS_VIENEU = False


def speech_like(seconds: float, amp: float = 0.3, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Tín hiệu giống tiếng nói: các âm tiết 180 ms có bao biên, ngăn bởi khoảng nghỉ ngắn."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    carrier = np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 360 * t) + 0.05 * rng.standard_normal(n)
    syllable = 0.5 - 0.5 * np.cos(2 * np.pi * (t % 0.22) / 0.22)
    return (amp * carrier * syllable).astype(np.float32)


class FakeTTS:
    """Giả lập VieneuEngine: 48 kHz, ~14 ký tự/giây; chunk trong `bad` câm ở lần sinh đầu."""

    sample_rate = 48000
    backend = "fake"

    def __init__(self, bad=()):
        self.bad = set(bad)
        self.calls = []

    def sampling(self):
        return {"temperature": 0.8}

    def synthesize_chunks(self, chunks):
        self.calls.append(list(chunks))
        out = []
        for i, text in enumerate(chunks):
            body = speech_like(max(0.5, len(text) / 14.0), seed=len(self.calls) * 100 + i, sr=self.sample_rate)
            out.append(np.concatenate([np.zeros(int(0.05 * self.sample_rate), np.float32), body]))
        return out, 0.01


class FakeASR:
    """Nghe ra đúng văn bản, trừ chunk trong `bad` ở lần nghe đầu (giả lập đọc vấp)."""

    model_id = "fake"

    def __init__(self, enabled=False, bad=(), chunks=None):
        self.enabled = enabled
        self.bad = set(bad)
        self.heard = {}
        self.chunks = chunks or {}

    def transcribe_batch(self, items):
        out = []
        for samples, _ in items:
            text = self.chunks.get(samples.size, "")
            seen = self.heard.get(text, 0)
            self.heard[text] = seen + 1
            out.append("ừ ừ ừ" if text in self.bad and seen == 0 else text)
        return out

    def unload(self):
        pass


class AudioUtilsTests(unittest.TestCase):
    def test_loud_float_is_not_treated_as_int16(self):
        # Bug cũ: đỉnh > 1.5 bị chia 32768 -> câu gần như câm.
        loud = np.array([0.0, 1.6, -1.7, 0.5], dtype=np.float32)
        self.assertAlmostEqual(1.7, float(np.max(np.abs(to_mono_float(loud)))), places=5)
        pcm = np.array([0, 16384, -32768], dtype=np.int16)
        np.testing.assert_allclose(to_mono_float(pcm), [0.0, 0.5, -1.0])

    def test_write_wav_scales_instead_of_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            wave = 1.5 * np.sin(np.linspace(0, 40 * np.pi, SR)).astype(np.float32)
            write_wav(path, wave, SR)
            data, _ = read_wav(path)
        self.assertLessEqual(float(np.max(np.abs(data))), 1.0)
        self.assertLess(np.mean(np.abs(data) > 0.995 * np.max(np.abs(data))), 0.08)


class CheckTests(unittest.TestCase):
    def test_cer_ignores_names(self):
        ref = "Kế Duyên mở cửa bước ra ngoài"
        hyp = "kế dzuyên mở cửa bước ra ngoài"
        self.assertGreater(char_error_rate(ref, hyp), 0.0)
        self.assertEqual(0.0, char_error_rate(ref, hyp, ["Kế Duyên", "kế dzuyên"]))

    def test_repeated_words(self):
        self.assertTrue(repeated_words("hắn đứng đứng dậy đi ra", "Hắn đứng dậy đi ra."))
        self.assertFalse(repeated_words("rất lâu rất lâu", "Rất lâu rất lâu."))


class TextTests(unittest.TestCase):
    def test_clean_drops_non_story_lines(self):
        raw = ("Chương 1: Mở đầu\r\n\r\nTác giả: Ai đó\n***\nHắn c·hết  rồi.\n\n“Ngươi đi đâu?”\n"
               "Bảng chuẩn hóa danh xưng\n- Lâm Viễn: nhân vật chính\n")
        self.assertEqual(["Chương 1: Mở đầu", "Hắn chết rồi.", "“Ngươi đi đâu?”"], clean_paragraphs(raw))

    def test_glossary_whole_words_longest_first(self):
        glossary = {"km": "ki lô mét", "km/h": "ki lô mét trên giờ", "Lam": "Làm"}
        self.assertEqual("chạy 60 ki lô mét trên giờ ở Làm Tinh",
                         apply_glossary("chạy 60 km/h ở Lam Tinh", glossary))
        self.assertEqual("Lamborghini", apply_glossary("Lamborghini", glossary))   # không thay giữa từ

    def test_chapter_text_one_paragraph_per_line(self):
        self.assertEqual("A b.\nC d.", chapter_text("A b.\n\n\nC d.\n"))


class VoiceRefTests(unittest.TestCase):
    def _clip(self):
        sr = 24000
        gap = np.zeros(int(sr * 0.4), dtype=np.float32)
        parts = []
        for i in range(5):                       # 5 câu 3 s, ngừng 0,4 s: ~16,6 s
            parts += [speech_like(3.0, seed=i), gap]
        return np.concatenate([gap] + parts), sr

    def test_segments_end_at_pauses_and_fit_vieneu(self):
        from core import voice_ref

        x, sr = self._clip()
        cands = voice_ref.candidate_segments(x, sr)
        self.assertTrue(cands)
        pauses = [t for t, _, _ in voice_ref.find_pauses(x, sr)]
        for a, b in cands:
            self.assertLessEqual(b - a, voice_ref.MAX_SECONDS)
            self.assertGreaterEqual(b - a, voice_ref.MIN_SECONDS)
            # Hai đầu đoạn rơi đúng chỗ ngừng (hoặc đầu file im lặng), không cắt giữa câu.
            self.assertTrue(a == 0.0 or min(abs(a - p) for p in pauses) < 0.05)
            self.assertTrue(min(abs(b - p) for p in pauses) < 0.05)
            cut = voice_ref.segment(x, sr, a, b)
            self.assertLess(cut.size / sr, 8.0)

    def test_short_clip_used_whole(self):
        from core import voice_ref

        x, sr = self._clip()
        short = x[: int(sr * 6)]
        self.assertEqual([(0.0, short.size / sr)], voice_ref.candidate_segments(short, sr))


class BuilderTests(unittest.TestCase):
    TEXT = "Hắn đứng dậy, đi ra cửa. Trời mưa rất to.\nKhông ai trả lời."

    def _builder(self, tmp, asr):
        return ChapterBuilder(FakeTTS(), asr, {"max_chars": 256}, {"max_cer": 0.12, "max_attempts": 3},
                              {"lufs": -14, "true_peak_db": -1.0})

    @unittest.skipUnless(HAS_VIENEU, "cần gói vieneu")
    def test_build_masters_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = self._builder(tmp, FakeASR(enabled=False))
            out = os.path.join(tmp, "ch.wav")
            timeline = builder.build(self.TEXT, "v", out, os.path.join(tmp, "t.json"))
            data, sr = read_wav(out)
            self.assertEqual(48000, sr)
            self.assertAlmostEqual(timeline["total_duration_sec"], data.size / sr, delta=0.05)
            if HAS_FFMPEG:
                stats = measure_loudness(out)
                self.assertAlmostEqual(-14.0, stats["input_i"], delta=1.0)
            calls = len(builder.tts.calls)
            builder.build(self.TEXT, "v", out, os.path.join(tmp, "t.json"))
            self.assertEqual(calls, len(builder.tts.calls))     # lần 2 dùng cache

    def test_asr_regenerates_only_bad_chunks(self):
        chunks = ["Hắn đứng dậy, đi ra cửa.", "Trời mưa rất to.", "Không ai trả lời."]
        tts = FakeTTS()
        asr = FakeASR(enabled=True, bad=[chunks[1]])
        builder = ChapterBuilder(tts, asr, {}, {"max_cer": 0.12, "max_attempts": 3}, {})
        orig = tts.synthesize_chunks

        def synth(texts):                       # nhớ độ dài audio -> văn bản để FakeASR "nghe" lại
            wavs, sec = orig(texts)
            for t, w in zip(texts, wavs):
                asr.chunks[w.size] = t
            return wavs, sec

        tts.synthesize_chunks = synth
        res = builder.render(chunks)
        self.assertEqual([chunks, [chunks[1]]], tts.calls)    # chỉ đọc lại chunk lỗi
        self.assertFalse(res["first_checks"][1]["ok"])
        self.assertTrue(all(c["ok"] for c in res["checks"]))
        self.assertIsNot(res["wavs"][1], res["first_wavs"][1])  # bản tốt thay bản lỗi
        self.assertIs(res["wavs"][0], res["first_wavs"][0])


@unittest.skipUnless(HAS_FFMPEG, "cần ffmpeg")
class VideoTests(unittest.TestCase):
    def test_episode_loops_background_to_audio_length(self):
        from core.video_maker import VideoMaker, media_duration, probe

        with tempfile.TemporaryDirectory() as tmp:
            bg = os.path.join(tmp, "bg.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=4",
                            "-pix_fmt", "yuv420p", bg], check=True)
            intro = os.path.join(tmp, "intro.mp3")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=500:d=1.5",
                            "-c:a", "libmp3lame", intro], check=True)
            wavs = []
            for i, seconds in enumerate((7.0, 5.5)):
                path = os.path.join(tmp, f"Chuong_{i + 1:04d}.wav")
                sf.write(path, speech_like(seconds, seed=i), SR, subtype="PCM_24")
                wavs.append(path)
            maker = VideoMaker({"fps": 25, "loop_crossfade_sec": 0.5, "x264_preset": "ultrafast",
                                "insert_gap_sec": 0.4}, os.path.join(tmp, "work"))
            out = os.path.join(tmp, "tap.mp4")
            total = maker.make_episode(wavs, bg, out, [{"after": 0, "file": intro}, {"after": 5, "file": intro}])
            # intro 1.5 + gap 0.4 + 7 + 5.5 ("after 5" không có chương phía sau -> không chèn)
            self.assertAlmostEqual(14.4, total, delta=0.1)
            self.assertAlmostEqual(total, media_duration(out), delta=0.15)
            audio = probe(out, "stream=codec_name,sample_rate,channels", "a:0").split()
            self.assertEqual(["aac", "48000", "2"], audio)
            video_dur = float(probe(out, "stream=duration", "v:0") or 0)
            self.assertAlmostEqual(total, video_dur, delta=0.15)


if __name__ == "__main__":
    unittest.main()


class DriveSyncTests(unittest.TestCase):
    def test_pulls_chapters_and_skips_uploaded_episodes(self):
        import argparse
        import shutil as sh

        import studio
        from core import drive

        with tempfile.TemporaryDirectory() as tmp:
            remote_in = os.path.join(tmp, "remote_chuong")
            os.makedirs(remote_in)
            for n in range(1, 16):
                with open(os.path.join(remote_in, f"Chuong_{n:04d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Chương {n}\n\nNội dung.")
            calls = []

            def fake_pull(remote, local, conf, includes=None):
                calls.append(remote)
                sh.copytree(remote, local, dirs_exist_ok=True)
                return True

            orig_pull, orig_list = drive.pull, drive.list_files
            drive.pull = fake_pull
            drive.list_files = lambda remote, conf: ["Chuong_0001-0010_Tap001.mp4", "ghi_chu.txt"]
            orig_data = studio.DATA
            studio.DATA = tmp
            try:
                cfg = {
                    "sync": {"enabled": True, "input": remote_in},
                    "upload": {"remote_dir": "gdrive:video"},
                    "episode": {"chapters_per_episode": 10},
                }
                args = argparse.Namespace(range=None, force=False, finish=False)
                st = studio.Studio(cfg, args)
                st.sync_from_drive(force=True)
                self.assertEqual(15, len(st.chapter_files()))
                pending = [n for n, _ in st.pending_chapters(set())]
                self.assertEqual(list(range(11, 16)), pending)   # tập 1 đã có trên Drive
                st.sync_from_drive()   # chưa tới chu kỳ -> không tải lại
                self.assertEqual(1, len(calls))
            finally:
                drive.pull, drive.list_files = orig_pull, orig_list
                studio.DATA = orig_data

    def test_uploaded_episode_names(self):
        from core.drive import uploaded_episodes

        names = ["Chuong_0001-0010_Tap001.mp4", "Chuong_1491-1496_Tap150.mp4", "x.mp4", "Chuong_0011-0020_Tap002.mp4.part"]
        self.assertEqual({"Chuong_0001-0010_Tap001.mp4", "Chuong_1491-1496_Tap150.mp4"}, uploaded_episodes(names))


class MultiJobTests(unittest.TestCase):
    def _studio(self, tmp, job, extra_cfg=None):
        import argparse

        import studio

        studio.DATA = studio.BASE = tmp
        cfg = {"episode": {"chapters_per_episode": 10, "watch_seconds": 0}}
        cfg.update(extra_cfg or {})
        args = argparse.Namespace(range=None, force=False, finish=False, stop_on_error=False, job=job, watch=True)
        return studio.Studio(cfg, args)

    def _chapters(self, tmp, count):
        os.makedirs(os.path.join(tmp, "input"), exist_ok=True)
        for n in range(1, count + 1):
            with open(os.path.join(tmp, "input", f"Chuong_{n:04d}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Chương {n}")

    def test_switch_to_20_chapters_keeps_unfinished_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._chapters(tmp, 45)
            s = self._studio(tmp, "audio1", {"episode": {"chapters_per_episode": 20, "watch_seconds": 0}})
            os.makedirs(s.video_dir, exist_ok=True)
            # Tập cũ 10 chương đã upload: không được che mất chương 11-20 của tập 20 chương mới.
            open(os.path.join(s.video_dir, "Chuong_0001-0010_Tap001.mp4.done"), "a").close()
            self.assertFalse(s.episode_done(1))
            open(os.path.join(s.video_dir, "Chuong_0001-0020_Tap001.mp4.done"), "a").close()
            self.assertTrue(s.episode_done(1))
            # Tập lẻ cuối (41-45) phủ tới chương cuối hiện có -> xong.
            open(os.path.join(s.video_dir, "Chuong_0041-0045_Tap003.mp4.done"), "a").close()
            self.assertTrue(s.episode_done(3))
            pending = [n for n, _ in s.pending_chapters(set())]
            self.assertEqual(list(range(21, 41)), pending)

    def test_lock_claim_and_stale(self):
        import time as _time

        from core import locks

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ch0001.lock")
            self.assertTrue(locks.claim(path, "A"))
            self.assertFalse(locks.claim(path, "B"))
            old = _time.time() - 4 * 3600
            os.utime(path, (old, old))
            self.assertTrue(locks.claim(path, "B", stale_sec=3 * 3600))   # khóa quá hạn -> giành lại

    def test_two_jobs_split_chapters_without_overlap(self):
        import threading as th
        import time as _time

        import studio

        with tempfile.TemporaryDirectory() as tmp:
            self._chapters(tmp, 12)
            done = []
            guard = th.Lock()

            def make(job):
                st = self._studio(tmp, job)
                st.voice = lambda: {"ref_audio": None, "fp": "x"}

                def fake_process(number, path, voice, glossary, names):
                    _time.sleep(0.05)
                    with guard:
                        done.append((number, job))
                    open(st.chapter_wav(number), "wb").close()

                st.process_chapter = fake_process
                return st

            orig = studio.DATA
            try:
                jobs = [make("A"), make("B")]
                threads = [th.Thread(target=j.run_audio, args=(False,)) for j in jobs]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            finally:
                studio.DATA = studio.BASE = orig
            numbers = sorted(n for n, _ in done)
            self.assertEqual(list(range(1, 13)), numbers)          # mỗi chương đúng 1 lần
            self.assertEqual({"A", "B"}, {j for _, j in done})     # cả 2 job cùng làm

    def test_video_watcher_builds_full_episodes_then_stops(self):
        import studio

        with tempfile.TemporaryDirectory() as tmp:
            self._chapters(tmp, 15)
            orig = studio.DATA
            try:
                st = self._studio(tmp, "video")
                for n in range(1, 16):
                    open(st.chapter_wav(n), "wb").close()
                built = []

                def fake_make(episode):
                    built.append(episode)
                    numbers, out = st.episode_info(episode)
                    open(out + ".done", "a").close()

                st._make_episode = fake_make
                st.watch_video()   # không còn job audio nào -> ghép tập đủ chương rồi dừng
            finally:
                studio.DATA = studio.BASE = orig
            self.assertEqual([1], built)   # tập 2 mới có 5/10 chương: chưa ghép (cần --finish)

    def test_project_config_and_shared_files(self):
        import studio

        cfg = studio.fill_project({"sync": {"input": "gdrive:Dự án truyện/{project}/chuong",
                                            "assets": {"Kenh": "gdrive:{project}/Kenh"}}}, "Giang_Than")
        self.assertEqual("gdrive:Dự án truyện/Giang_Than/chuong", cfg["sync"]["input"])
        self.assertEqual("gdrive:Giang_Than/Kenh", cfg["sync"]["assets"]["Kenh"])
        with tempfile.TemporaryDirectory() as tmp:
            orig_data, orig_base = studio.DATA, studio.BASE
            try:
                studio.BASE = tmp
                studio.DATA = os.path.join(tmp, "projects", "A")
                os.makedirs(studio.DATA)
                open(os.path.join(tmp, "rclone.conf"), "w").close()
                # File chung ở thư mục gốc được dùng khi bộ truyện không có bản riêng
                self.assertEqual(os.path.join(tmp, "rclone.conf"), studio.rel("rclone.conf"))
                self.assertEqual(os.path.join(studio.DATA, "input"), studio.rel("input"))
            finally:
                studio.DATA, studio.BASE = orig_data, orig_base


class ProjectNameTests(unittest.TestCase):
    def test_match_folder_without_diacritics(self):
        from core.drive import match_folder

        names = ["Đại Niết Bàn", "Trọng sinh 2003: Giang Thần", "Sample"]
        self.assertEqual("Đại Niết Bàn", match_folder("Đại Niết Bàn", names))
        self.assertEqual("Đại Niết Bàn", match_folder("Dai Niet Ban", names))
        self.assertEqual("Đại Niết Bàn", match_folder("dai_niet_ban", names))
        self.assertEqual("Trọng sinh 2003: Giang Thần", match_folder("Trong sinh 2003: Giang Than", names))
        self.assertIsNone(match_folder("i Nit Bn", names))   # chữ bị rơi dấu -> không đoán bừa

    def test_project_data_dirs_never_fall_back_to_shared(self):
        import argparse

        import studio

        with tempfile.TemporaryDirectory() as tmp:
            studio.BASE = tmp
            for d in ("input", "output/audio", "assets/voice"):
                os.makedirs(os.path.join(tmp, d))
            studio.DATA = os.path.join(tmp, "projects", "Đại Niết Bàn")
            os.makedirs(studio.DATA)
            args = argparse.Namespace(range=None, force=False, finish=False, stop_on_error=False, job=None, watch=False)
            s = studio.Studio({"episode": {"chapters_per_episode": 20}}, args)
            self.assertTrue(s.input_dir.startswith(studio.DATA))
            self.assertTrue(s.audio_dir.startswith(studio.DATA))
            self.assertTrue(studio.own("assets/voice").startswith(studio.DATA))
