import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from core.asr_checker import char_error_rate, repeated_words
from core.audio_builder import AudioBuilder
from core.audio_master import master_wav, measure_loudness
from core.audio_utils import (
    longest_internal_gap_ms,
    normalize_voiced,
    read_wav,
    room_tone,
    to_mono_float,
    voiced_rms_db,
    write_wav,
)
from core.text_processor import build_chunks, clean_paragraphs
from core.vi_normalizer import normalize_for_tts

SR = 24000
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def speech_like(seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """Tín hiệu giống tiếng nói: các âm tiết 180 ms có bao biên, ngăn bởi khoảng nghỉ ngắn."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    carrier = np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 360 * t) + 0.05 * rng.standard_normal(n)
    syllable = 0.5 - 0.5 * np.cos(2 * np.pi * (t % 0.22) / 0.22)
    return (amp * carrier * syllable).astype(np.float32)


class FakeTTS:
    """Giả lập OmniVoice: ~14 ký tự/giây, có khoảng lặng hai đầu; có thể hỏng ở lượt đầu."""

    def __init__(self, fail_first=()):
        self.sample_rate = SR
        self.fail_first = set(fail_first)
        self.calls = []

    def synthesize_batch(self, texts, ref_audio, ref_text, seed):
        self.calls.append((list(texts), seed))
        out = []
        for i, text in enumerate(texts):
            if text in self.fail_first and seed == 1234:
                out.append(np.zeros(int(0.1 * SR), dtype=np.float32))  # câm -> phải đọc lại
                continue
            body = speech_like(max(0.5, len(text) / 14.0), amp=0.1 + 0.3 * (i % 3) / 2, seed=i)
            pad = np.zeros(int(0.3 * SR), dtype=np.float32)
            out.append(np.concatenate([pad, body, pad]))
        return out

    def unload(self):
        pass


class FakeASR:
    enabled = False

    def transcribe_batch(self, items):
        return [None] * len(items)

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
        # Không bị cắt phẳng đỉnh: sine nguyên vẹn có ~6.4% mẫu sát đỉnh, bản bị cắt cứng ~50%.
        self.assertLess(np.mean(np.abs(data) > 0.995 * np.max(np.abs(data))), 0.08)

    def test_room_tone_level(self):
        tone = room_tone(1000, SR, -66.0)
        rms_db = 20 * np.log10(np.sqrt(np.mean(tone[2000:-2000] ** 2)))
        self.assertAlmostEqual(-66.0, rms_db, delta=2.0)
        self.assertEqual(0.0, float(np.max(np.abs(room_tone(500, SR, None)))))

    def test_voiced_levels_are_matched(self):
        quiet = np.concatenate([np.zeros(SR), speech_like(2, amp=0.05), np.zeros(SR)])
        loud = speech_like(2, amp=0.5, seed=3)
        a = voiced_rms_db(normalize_voiced(quiet, SR, -20, 15), SR)
        b = voiced_rms_db(normalize_voiced(loud, SR, -20, 15), SR)
        self.assertAlmostEqual(a, b, delta=0.5)

    def test_internal_gap(self):
        x = np.concatenate([speech_like(1), np.zeros(int(1.5 * SR), dtype=np.float32), speech_like(1)])
        self.assertGreaterEqual(longest_internal_gap_ms(x, SR), 1400)


class PitchTests(unittest.TestCase):
    def test_f0_tracks_known_pitch(self):
        from core.audio_utils import median_f0

        for hz in (110.0, 180.0, 240.0):
            t = np.arange(int(1.5 * SR)) / SR
            x = (0.3 * np.sin(2 * np.pi * hz * t) + 0.1 * np.sin(4 * np.pi * hz * t)).astype(np.float32)
            self.assertAlmostEqual(hz, median_f0(x, SR), delta=hz * 0.03)

    def test_off_pitch_sentence_is_flagged(self):
        builder = AudioBuilder(FakeTTS(), FakeASR(), {}, {"max_pitch_dev_st": 1.5}, tempfile.gettempdir())
        builder.ref_f0 = 180.0
        text = "Hắn đứng dậy đi ra ngoài cửa sổ nhìn trời."
        t = np.arange(int(2 * SR)) / SR
        on = (0.3 * np.sin(2 * np.pi * 182 * t)).astype(np.float32)
        off = (0.3 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)   # +4,2 bán cung
        ok = builder._signal_check(text, on, on, SR)
        bad = builder._signal_check(text, off, off, SR)
        self.assertFalse(any("lệch giọng" in p for p in ok["problems"]))
        self.assertTrue(any("lệch giọng" in p for p in bad["problems"]))
        self.assertLess(builder._score(builder._asr_check(text, None, ok)),
                        builder._score(builder._asr_check(text, None, bad)))


class CheckTests(unittest.TestCase):
    def test_cer_ignores_names(self):
        ref = "Kế Duyên mở cửa bước ra ngoài"
        hyp = "kế dzuyên mở cửa bước ra ngoài"
        self.assertGreater(char_error_rate(ref, hyp), 0.0)
        self.assertEqual(0.0, char_error_rate(ref, hyp, ["Kế Duyên", "kế dzuyên"]))

    def test_repeated_words(self):
        self.assertTrue(repeated_words("hắn đứng đứng dậy đi ra", "Hắn đứng dậy đi ra."))
        self.assertFalse(repeated_words("rất lâu rất lâu", "Rất lâu rất lâu."))

    def test_split_half(self):
        left, right = AudioBuilder.split_half("Hắn đứng dậy, nhìn ra ngoài cửa sổ, trời đã tối hẳn rồi mà vẫn chưa thấy ai.")
        self.assertTrue(left.endswith(","))
        self.assertTrue(right)


class BuilderTests(unittest.TestCase):
    def _chunks(self):
        raw = "Chương 1: Mở đầu\n\nHắn đứng dậy, đi ra cửa. Trời mưa rất to.\n\n“Ngươi đi đâu?”\n\nKhông ai trả lời."
        return build_chunks(clean_paragraphs(raw), max_chars=220)

    def test_build_retries_and_masters(self):
        chunks = self._chunks()
        bad = chunks[1]["tts"]
        with tempfile.TemporaryDirectory() as tmp:
            tts = FakeTTS(fail_first=[bad])
            builder = AudioBuilder(tts, FakeASR(), {"batch_size": 4}, {"max_attempts": 3}, tmp,
                                   {"lufs": -14, "true_peak_db": -1.0})
            out = os.path.join(tmp, "ch.wav")
            timeline = builder.build(chunks, None, None, "v", out, os.path.join(tmp, "t.json"))
            self.assertTrue(os.path.exists(out))
            # Câu hỏng ở lượt đầu được đọc lại với seed khác, không bị bỏ qua.
            self.assertTrue(any(bad in texts and seed != 1234 for texts, seed in tts.calls))
            self.assertEqual([], timeline["warnings"])
            starts = [c["start_ms"] for c in timeline["chunks"]]
            self.assertEqual(sorted(starts), starts)
            self.assertTrue(all(c["end_ms"] > c["start_ms"] for c in timeline["chunks"]))
            data, sr = read_wav(out)
            self.assertAlmostEqual(timeline["total_duration_sec"], data.size / sr, delta=0.05)
            if HAS_FFMPEG:
                stats = measure_loudness(out)
                self.assertAlmostEqual(-14.0, stats["input_i"], delta=1.0)
                self.assertLessEqual(stats["input_tp"], -1.0)
            # Lần 2: dùng cache, không gọi TTS nữa.
            calls = len(tts.calls)
            builder.build(chunks, None, None, "v", out, os.path.join(tmp, "t.json"))
            self.assertEqual(calls, len(tts.calls))

    def test_batches_respect_limits(self):
        builder = AudioBuilder(FakeTTS(), FakeASR(), {"batch_size": 3, "batch_max_chars": 50}, {}, tempfile.gettempdir())
        texts = ["a" * 10, "b" * 40, "c" * 5, "d" * 20, "e" * 30]
        batches = builder._batches(texts)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(5)))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(len(batch) == 1 or sum(len(texts[i]) for i in batch) <= 50)

    def test_assemble_keeps_all_speech(self):
        builder = AudioBuilder(FakeTTS(), FakeASR(), {}, {}, tempfile.gettempdir())
        parts = [speech_like(1.0, seed=i) for i in range(3)]
        rendered = [({"id": i, "pause_after_ms": p}, parts[i], {"ok": True}) for i, p in enumerate((150, 600, 800))]
        full, entries, _ = builder.assemble(rendered, SR)
        # pause 150 < tail+lead -> chồng phần lặng; pause 600 -> chèn room tone 340 ms.
        expected = sum(p.size for p in parts) - int(SR * 0.08) + int(SR * 0.34) + int(SR * 0.54)
        self.assertAlmostEqual(expected, full.size, delta=SR * 0.01)
        self.assertEqual(entries[1]["start_ms"], 1000 - 80)


class TextTests(unittest.TestCase):
    def test_normalizer(self):
        text = normalize_for_tts("【LV1: Tốc độ +20% ×50】 năm 1998")
        self.assertNotRegex(text, r"[\d【】%×]")
        self.assertIn("một nghìn chín trăm chín mươi tám", text)


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
                st.voice = lambda: {"ref_audio": None, "ref_text": None, "fp": "x"}

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
