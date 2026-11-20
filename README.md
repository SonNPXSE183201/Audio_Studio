# 🎙️ Audio Studio

Biến chương truyện chữ tiếng Việt thành **video đọc truyện** đăng YouTube:
- giọng đọc AI clone **mượt, rõ, không vấp, không sướt, không mất câu**;
- tự nghe lại từng câu và đọc lại câu bị vấp, nuốt chữ hay lặp từ;
- ghép mỗi tập 10 chương, chèn intro/watermark của kênh;
- lặp một video nền ngắn (1–2 phút) cho đến hết audio;
- đẩy lên Google Drive.

Chạy bằng terminal trên RunPod (hoặc máy có GPU NVIDIA). Đọc audio bằng GPU, ghép video bằng CPU **chạy song song**, nên pod không lúc nào ngồi chờ.

**Bạn chỉ cần thả file chương `.txt` vào `input/`, tool tự làm phần còn lại.**

> Hướng dẫn từng bước trên RunPod, bảng **thời gian và chi phí theo từng loại GPU**: [docs/HUONG_DAN_RUNPOD.md](docs/HUONG_DAN_RUNPOD.md)
>
> Ước tính nhanh: 1 tập 10 chương (~80 phút audio) mất **~7–20 phút** và **~$0.05–0.18** tùy GPU. Cả bộ 800 chương khoảng **$4–14**.

## Quy trình tự động

| Bước | Tool làm gì | Tự kiểm tra |
|---|---|---|
| Làm sạch | Bỏ bảng chuẩn hóa danh xưng, dòng "NGUYÊN TÁC/Thời gian", `***`; sửa dấu câu dính chữ; `c·hết` → `chết` | — |
| Chuẩn hóa để đọc | Số (`1.500` → một nghìn năm trăm), %, giờ, đơn vị, tiền tệ, `Lv`, `【】`, từ mượn (hack, game, VIP…) | cách đọc riêng trong `input/glossary.json` |
| Chia câu | Mỗi đoạn đọc là **câu trọn vẹn** (≤ 220 ký tự); chỉ cắt ở dấu phẩy khi câu quá dài | nhịp nghỉ theo dấu câu, đoạn văn, lời thoại |
| Đọc | OmniVoice clone giọng mẫu; **giọng mẫu mã hóa 1 lần**, **nhiều câu sinh cùng lúc** (batch theo VRAM) | — |
| Nghe lại | Kiểm tra tốc độ đọc, khoảng câm giữa câu, đuôi câu cụt; **PhoWhisper nghe lại theo batch**: sai > 12% ký tự hoặc lặp từ thì đọc lại (seed khác, tối đa 3 lượt) | vẫn lỗi thì **tách đôi câu** đọc từng nửa; không bao giờ bỏ qua câu |
| Nối câu | Cắt lặng nhẹ tay (giữ phụ âm đầu, đuôi thanh điệu), **cân bằng độ to từng câu**, fade mềm, khoảng nghỉ có "không khí phòng" thay vì im lặng tuyệt đối | không có tiếng click, câu không to nhỏ lệch nhau |
| Hậu kỳ | Lọc ù 70 Hz, chuẩn **−14 LUFS** (chuẩn YouTube) bằng gain cố định + limiter chặn đỉnh −1 dBTP | không "phồng xẹp", không vỡ tiếng |
| Ghép tập | Intro + chương 1–4 + watermark + 5–8 + watermark + 9–10, nối WAV không nén (chính xác từng mẫu) | không hụt tiếng ở chỗ nối |
| Video | Video nền chuẩn hóa **1 lần** thành đơn vị lặp (crossfade cuối→đầu để lặp liền mạch), lặp bằng concat copy; audio nén **AAC 192k 48 kHz stereo đúng 1 lần** | độ dài hình = độ dài tiếng (sai < 0,15 s) |
| Đóng gói | Đẩy lên Drive bằng rclone, thành công thì xóa video và để lại `.done` | lỗi mạng thì giữ file, lần sau thử lại |

Mọi bước có cache: pod tắt giữa chừng thì lần sau chạy tiếp từ câu đang đọc dở (lưu sau mỗi 64 câu).

## Vì sao giọng không còn sướt / mất tiếng (so với DocTruyenV2)

| Lỗi cũ | Hậu quả | Audio Studio |
|---|---|---|
| Câu TTS lỗi bị **bỏ qua** | mất câu, truyện nhảy cóc | đọc lại → tách đôi → dừng chương nếu vẫn hỏng, không bao giờ xuất audio thiếu |
| Audio nén mất dữ liệu **3 lần** (MP3 chương → MP3 tập 128k → AAC 128k) | tiếng sướt, rè, xì | WAV không nén đến bước cuối, nén AAC 192k **1 lần** |
| Nối MP3 | hụt tiếng / click ở chỗ nối chương, intro | nối WAV chính xác từng mẫu |
| `-stream_loop -1 -c:v copy` trực tiếp | timestamp gãy ở chỗ lặp: khựng hình, có máy mất tiếng | đơn vị lặp chuẩn hóa + concat liên tục |
| Không chuẩn hóa số, ký hiệu | đọc vấp ở `1.500`, `20%`, `【】`, từ tiếng Anh | bộ chuẩn hóa tiếng Việt |
| Không nghe lại | câu vấp, nuốt chữ lọt vào video | ASR nghe lại + đọc lại |
| Câu to câu nhỏ, im lặng tuyệt đối giữa câu | nghe giật cục | cân bằng độ to, room tone, fade mềm |
| Mỗi câu mã hóa lại giọng mẫu, sinh từng câu một | chậm, tốn tiền pod | mã hóa 1 lần, sinh theo batch |

## Chuẩn bị dữ liệu

```
input/
  Chuong_0001.txt, Chuong_0002.txt ...   số chương lấy từ tên file
  glossary.json   (tùy chọn) cách đọc riêng: {"Lv": "cấp", "QQ": "qui qui", "Tô Vũ": "Tô Vũ"}
  names.txt       (tùy chọn) mỗi dòng 1 tên riêng: ASR không tính sai ở các tên này
assets/voice/
  voice_reference_clean.wav   giọng mẫu 6–20 giây, sạch, không nhạc nền
  voice_reference_clean.txt   lời đọc KHỚP CHÍNH XÁC với giọng mẫu (thiếu thì ASR tự nghe ra)
assets/background/
  background.mp4              video nền ngắn, lặp đến hết audio
Kenh/
  Script_Intro.mp3, Script_Watermark.mp3, Script_Watermark1.mp3   (tùy chọn)
```

Giọng mẫu càng sạch thì giọng đọc càng sạch: không tiếng vang, không nhạc, không tiếng ồn. Lời trong file `.txt` phải khớp từng chữ với giọng mẫu, nếu sai thì giọng đọc dễ vấp ở đầu câu.

## Lệnh thường dùng

```bash
bash run.sh check                    # kiểm tra ffmpeg, GPU, giọng mẫu, video nền, số chương
bash run.sh bench --price 0.69       # đo tốc độ thật của pod (~5 phút) -> thời gian và $ mỗi chương/tập
bash run.sh start                    # chạy nền: đọc audio + tự ghép tập khi đủ 10 chương
bash run.sh start --range 1-400      # chỉ xử lý chương 1-400
bash run.sh log                      # xem tiến độ (Ctrl+C để thoát xem, tool vẫn chạy)
bash run.sh stop                     # dừng

bash run.sh audio                    # chỉ đọc audio
bash run.sh video                    # chỉ ghép các tập đã đủ audio
bash run.sh video --finish           # gom cả tập lẻ cuối truyện (vd. chương 801-806)
bash run.sh audio --range 15-15 --force   # đọc lại chương 15
```

Trên máy Windows có GPU: `python studio.py check`, `python studio.py run`… (cần ffmpeg trong PATH và Python có torch CUDA + omnivoice).

## Cấu trúc thư mục

### Mã nguồn (trên GitHub)

```
Audio_Studio/
├── studio.py               Điều phối: đọc chương → audio → ghép tập → Drive (GPU và CPU chạy song song)
├── config.yaml             Cấu hình mặc định (trên pod được chép sang thư mục dữ liệu để sửa)
├── run.sh                  check / start / log / stop / run / audio / video trên pod
├── requirements.txt
├── pod/
│   ├── bootstrap.sh        Một lệnh cho pod mới/vừa Start: clone code + cài đặt
│   └── setup_pod.sh        Cài ffmpeg, rclone, omnivoice; tải trước model lên volume
├── core/
│   ├── text_processor.py   Làm sạch chương, chia câu, nhịp nghỉ theo dấu câu
│   ├── vi_normalizer.py    Chuẩn hóa tiếng Việt để đọc (số, ký hiệu, từ mượn)
│   ├── tts_engine.py       OmniVoice: voice prompt dùng lại, sinh theo batch, tự chia batch khi hết VRAM
│   ├── asr_checker.py      PhoWhisper nghe lại theo batch, tỷ lệ sai ký tự (bỏ qua tên riêng), lặp từ
│   ├── audio_builder.py    Đọc theo lượt, đọc lại câu lỗi, tách đôi câu, nối mượt, cache
│   ├── audio_utils.py      Xử lý audio numpy: cắt lặng, fade, cân bằng độ to, room tone, kiểm tra tín hiệu
│   ├── audio_master.py     Hậu kỳ FFmpeg: lọc ù, chuẩn LUFS 2 lượt, limiter
│   ├── video_maker.py      Đơn vị lặp video nền, nối audio tập, xuất MP4
│   ├── drive.py            Upload rclone
│   └── utils.py            Log, JSON, fingerprint
├── assets/voice/           Giọng mẫu (file .wav không đưa lên GitHub)
├── docs/HUONG_DAN_RUNPOD.md
└── tests/                  Kiểm thử: python -m pytest tests
```

### Dữ liệu (trên pod: `/workspace/audio_studio`, không mất khi Stop pod)

```
/workspace/audio_studio/
├── config.yaml             Cấu hình bạn sửa (ưu tiên hơn bản trong code)
├── rclone.conf             Google Drive
├── input/                  Chương .txt, glossary.json, names.txt
├── assets/voice/           Giọng mẫu
├── assets/background/      Video nền
├── Kenh/                   Intro, watermark
├── output/audio/           Chuong_0001.wav … (WAV 24-bit đã hậu kỳ)
├── output/video/           Chuong_0001-0010_Tap001.mp4 (hoặc .done sau khi lên Drive)
├── work/
│   ├── tts/                Câu đã đọc (cache, tự xóa sau khi xong chương)
│   ├── chXXXX/timeline.json  Mốc thời gian từng câu + danh sách câu còn nghi lỗi (warnings)
│   ├── video/              Đơn vị lặp video nền, intro đã chuẩn hóa
│   ├── logs/               Log theo ngày
│   └── studio_run.log      Log chạy nền
└── .cache/huggingface/     Model OmniVoice, PhoWhisper (tải 1 lần)
```

## Tùy chỉnh thường gặp (`config.yaml`)

| Muốn | Sửa |
|---|---|
| To hơn / nhỏ hơn | `master.lufs`: −14 (mặc định, chuẩn YouTube), −13 to hơn, −16 nhỏ hơn |
| Nghỉ lâu hơn giữa câu, đoạn | `tts.pauses_ms` |
| Nhanh hơn (rẻ hơn) | `tts.num_step: 24` (nghe thử trước), `asr.model_id: vinai/PhoWhisper-medium`, hoặc `asr.enabled: false` |
| Đọc chậm/nhanh | `tts.speed` (nên giữ 1.0) |
| Một từ bị đọc sai | thêm vào `input/glossary.json`: `{"từ gốc": "cách đọc"}` |
| ASR báo sai ở tên nhân vật | thêm tên vào `input/names.txt` |
| Nghe lẹt xẹt | thử `tts.dtype: bf16` |
| Video nền giật ở chỗ lặp | tăng `video.loop_crossfade_sec` |
| Ép độ phân giải | `video.resolution: [1920, 1080]` |
| Số chương mỗi tập, intro/watermark | `episode.chapters_per_episode`, `episode.inserts` |
| Đổi giọng | `voice.ref_audio` + `voice.ref_text` (đổi giọng thì các câu đọc lại theo giọng mới) |

Sau khi chạy, mở `work/chXXXX/timeline.json` và xem mục `warnings`: đó là các câu vẫn nghi lỗi sau mọi lần đọc lại, kèm thời điểm (`time_sec`) để nghe kiểm tra nhanh.
