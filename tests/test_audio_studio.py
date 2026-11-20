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
                self.assertLessEqual(stats["input_tp"], -0.8)
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
