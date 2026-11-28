# 🎙️ Audio Studio

Biến chương truyện chữ tiếng Việt thành **video đọc truyện** đăng YouTube:
- giọng đọc AI clone **mượt, rõ, không vấp, không sướt, không mất câu**;
- tự nghe lại từng câu và đọc lại câu bị vấp, nuốt chữ hay lặp từ;
- ghép mỗi tập 40 chương, chèn intro/watermark của kênh;
- lặp một video nền ngắn (1–2 phút) cho đến hết audio;
- đẩy lên Google Drive.

Chạy bằng terminal trên RunPod (hoặc máy có GPU NVIDIA). Đọc audio bằng GPU, ghép video bằng CPU **chạy song song**, nên pod không lúc nào ngồi chờ.

**Bạn chỉ cần thả file chương `.txt` lên Google Drive, deploy pod bằng template `Audio Studio` rồi chạy 1 lệnh:** tool tự tải chương, đọc, ghép tập 40 chương và đẩy video về Drive.

```bash
curl -fsSL https://raw.githubusercontent.com/SonNPXSE183201/Audio_Studio/main/pod/bootstrap.sh \
  | AUTO_RUN=1 PROJECTS="Đại Niết Bàn" JOBS=2 bash
```

> Hướng dẫn từng bước trên RunPod, bảng **thời gian và chi phí theo từng loại GPU**: [docs/HUONG_DAN_RUNPOD.md](docs/HUONG_DAN_RUNPOD.md)
>
> Ước tính nhanh: 1 tập 20 chương (~160 phút audio) mất **~14–40 phút** và **~$0.10–0.36** tùy GPU. Cả bộ 800 chương khoảng **$4–14**.

## Quy trình tự động

| Bước | Tool làm gì | Tự kiểm tra |
|---|---|---|
| Làm sạch | Bỏ bảng chuẩn hóa danh xưng, dòng "NGUYÊN TÁC/Thời gian", `***`; sửa dấu câu dính chữ; `c·hết` → `chết` | — |
| Đọc | **VieNeu-TTS v3 Turbo** (48 kHz) theo đúng logic của VieNeu: sea-g2p chuẩn hóa số, ký hiệu, viết tắt và tách câu; gom câu thành chunk ≤ 256 ký tự; GPU sinh nhiều chunk một lượt; nghỉ 0,7 s giữa đoạn, 0,5 s giữa câu, 0,3 s trong câu | cách đọc riêng trong `chuong/glossary.json` (Drive) |
| Giọng mẫu | VieNeu chỉ lấy 8 s đầu của clip mẫu: tool tự cắt các **đoạn câu trọn vẹn 3–7,8 s** ở chỗ ngừng sạch, cho mỗi đoạn đọc thử vài câu của chương đầu, giữ đoạn có giọng sinh ra **giống giọng gốc nhất** (so hồ sơ giọng) | không cần file `.txt` lời mẫu |
| Nghe lại | PhoWhisper nghe lại từng chunk: sai > 12% ký tự hoặc lặp từ thì đọc lại (tối đa 3 lượt), giữ bản tốt nhất | chunk vẫn nghi lỗi ghi vào `timeline.json` |
| Hậu kỳ | Lọc ù 70 Hz, chuẩn **−14 LUFS** (chuẩn YouTube) bằng gain cố định + limiter chặn đỉnh −1 dBTP | không "phồng xẹp", không vỡ tiếng |
| Ghép tập | Intro + chương 1–16 + watermark + 17–32 + watermark + 33–40, nối WAV không nén (chính xác từng mẫu) | không hụt tiếng ở chỗ nối |
| Video | Video nền chuẩn hóa **1 lần** thành đơn vị lặp (crossfade cuối→đầu để lặp liền mạch), lặp bằng concat copy; audio nén **AAC 192k 48 kHz stereo đúng 1 lần** | độ dài hình = độ dài tiếng (sai < 0,15 s) |
| Lấy dữ liệu | Tải chương `.txt` + giọng mẫu, video nền, intro từ Google Drive; 20 phút xem lại một lần để lấy chương mới | tập đã có trên Drive thì bỏ qua, pod mới tự chạy tiếp |
| Đóng gói | Đẩy lên Drive bằng rclone, thành công thì xóa video và audio chương, để lại `.done` | lỗi mạng thì giữ file, lần sau thử lại |

Mỗi chương là một đơn vị: pod tắt giữa chừng thì lần sau đọc tiếp từ chương chưa có audio (GPU đọc 1 chương chỉ mất vài chục giây nên không cần cache từng câu).

## Vì sao giọng không còn sướt / mất tiếng (so với DocTruyenV2)

| Lỗi cũ | Hậu quả | Audio Studio |
|---|---|---|
| Câu TTS lỗi bị **bỏ qua** | mất câu, truyện nhảy cóc | chunk rỗng thì dừng chương, không bao giờ xuất audio thiếu |
| Audio nén mất dữ liệu **3 lần** (MP3 chương → MP3 tập 128k → AAC 128k) | tiếng sướt, rè, xì | WAV không nén đến bước cuối, nén AAC 192k **1 lần** |
| Nối MP3 | hụt tiếng / click ở chỗ nối chương, intro | nối WAV chính xác từng mẫu |
| `-stream_loop -1 -c:v copy` trực tiếp | timestamp gãy ở chỗ lặp: khựng hình, có máy mất tiếng | đơn vị lặp chuẩn hóa + concat liên tục |
| Không chuẩn hóa số, ký hiệu | đọc vấp ở `1.500`, `20%`, `【】`, từ tiếng Anh | sea-g2p của VieNeu |
| Không nghe lại | câu vấp, nuốt chữ lọt vào video | ASR nghe lại + đọc lại |
| Mỗi câu mã hóa lại giọng mẫu, sinh từng câu một | chậm, tốn tiền pod | hồ sơ giọng tạo 1 lần, GPU sinh 32 chunk một lượt |

## Chuẩn bị dữ liệu (Google Drive)

Mỗi bộ truyện là một thư mục trong **`Drive của tôi / Dự án truyện /`**. Tên thư mục chính là tên bộ truyện đưa vào `PROJECTS` (giữ nguyên dấu, dấu cách):

```
Dự án truyện/
├── Sample/                ← thư mục mẫu: copy rồi đổi tên thành tên bộ mới
├── Đại Niết Bàn/
│   ├── chuong/            Chuong_0001.txt, Chuong_0002.txt … (số chương lấy từ tên file)
│   │                      glossary.json (tùy chọn) cách đọc riêng: {"Lv": "cấp", "QQ": "qui qui"}
│   │                      names.txt     (tùy chọn) mỗi dòng 1 tên riêng: ASR không tính sai các tên này
│   ├── voice/             voice_reference_clean.wav  giọng mẫu 6–20 giây, sạch, không nhạc nền
│   │                      voice_reference_clean.txt  lời đọc KHỚP CHÍNH XÁC với giọng mẫu
│   ├── background/        background.mp4  video nền 1–2 phút, lặp đến hết audio
│   ├── Kenh/              Script_Intro.mp3 (đầu tập), Script_Watermark.mp3 (trước chương 9),
│   │                      Script_Watermark1.mp3 (trước chương 17); thiếu file nào thì bỏ qua chỗ đó
│   └── video/             tool tự đẩy Chuong_0001-0020_Tap001.mp4 … vào đây
└── <bộ khác>/ …
```

- **Thêm bộ mới:** copy `Sample`, đổi tên, thả file vào từng thư mục con, chạy pod với `PROJECTS="<tên y hệt trên Drive>"`.
- Tool **tự bật Drive** khi pod có `rclone.conf` (lấy từ RunPod Secret, xem [hướng dẫn mục 5](docs/HUONG_DAN_RUNPOD.md#5-dữ-liệu-tự-lấy-từ-google-drive-khuyên-dùng)). Cứ 20 phút tool xem Drive có chương mới không; tập đã có trong `video/` thì bỏ qua, nên pod mới tự chạy tiếp từ chỗ dở.
- Mẫu thư mục cũng có trong repo: `drive_template/Sample/` (kèm `HUONG_DAN.txt`).

Giọng mẫu càng sạch thì giọng đọc càng sạch: không tiếng vang, không nhạc, không tiếng ồn. Lời trong file `.txt` phải khớp từng chữ với giọng mẫu, nếu sai thì giọng đọc dễ vấp ở đầu câu.

## Nhiều bộ truyện, nhiều job

- **Mỗi bộ truyện một thư mục riêng**: chạy với `--project <tên>`. Dữ liệu nằm ở `projects/<tên>/` (chương, audio, video, cache), không bao giờ lẫn chương giữa các bộ.
- **Một `config.yaml` cho mọi bộ**: đường dẫn Drive viết `gdrive:Dự án truyện/{project}/chuong`; tool thay `{project}` bằng tên bộ (đặt tên trùng thư mục trên Drive). Bộ nào cần khác (giọng, video nền…) thì thêm `projects/<tên>/config.yaml`, chỉ ghi những dòng muốn đổi.
- **Nhiều job đọc audio cùng lúc tự chia chương**: job nào cũng lấy chương đầu tiên chưa có ai làm (khóa `work/locks/chXXXX.lock`), nên không bao giờ đọc trùng và không cần chia `--range`. Job chết giữa chừng thì job khác làm tiếp chương đó.
- **Một job video chạy ngầm** (CPU): mỗi phút quét audio, đủ 40 chương liên tiếp là ghép tập + upload; mỗi tập chỉ một job được ghép (khóa `tapNNN.lock`). Tự dừng khi các job audio đã xong.

```bash
bash run.sh start --project "Đại Niết Bàn" --jobs 2     # 2 job audio + 1 job video
bash run.sh status --project "Đại Niết Bàn"             # job đang chạy, số chương / audio / tập đã xong
bash run.sh log --project "Đại Niết Bàn"
bash run.sh stop --project "Đại Niết Bàn"

# Nhiều bộ, chạy lần lượt (bộ này xong mới sang bộ sau):
bash pod/run_projects.sh --jobs 2 "Đại Niết Bàn" "Trọng sinh 2003: Giang Thần"
# hoặc khi tạo pod: PROJECTS="Đại Niết Bàn;Trọng sinh 2003: Giang Thần"   (ngăn cách bằng dấu ;)
```

## Lệnh thường dùng

```bash
bash run.sh check                    # kiểm tra ffmpeg, GPU, giọng mẫu, video nền, số chương
bash run.sh bench --price 0.69       # đo tốc độ thật của pod (~5 phút) -> thời gian và $ mỗi chương/tập
bash run.sh start                    # chạy nền: đọc audio + tự ghép tập khi đủ 40 chương
bash run.sh start --range 1-400      # chỉ xử lý chương 1-400
bash run.sh log                      # xem tiến độ (Ctrl+C để thoát xem, tool vẫn chạy)
bash run.sh stop                     # dừng

bash run.sh audio                    # chỉ đọc audio
bash run.sh video                    # chỉ ghép các tập đã đủ audio
bash run.sh video --finish           # gom cả tập lẻ cuối truyện (vd. chương 801-806)
bash run.sh audio --range 15-15 --force   # đọc lại chương 15
```

Trên máy Windows có GPU: `python studio.py check`, `python studio.py run`… (cần ffmpeg trong PATH và `pip install -r requirements.txt`; không có torch CUDA thì VieNeu chạy CPU/ONNX, chậm hơn nhiều).

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
│   └── setup_pod.sh        Cài ffmpeg, rclone, torch 2.8 + vieneu; tải trước model lên volume
├── core/
│   ├── text_processor.py   Làm sạch file chương, glossary của bộ truyện
│   ├── voice_ref.py        Tìm đoạn câu trọn vẹn ≤ 7,8 s trong giọng mẫu (ứng viên cho VieNeu)
│   ├── vieneu_engine.py    VieNeu v3 Turbo: chunk + sinh + ghép theo logic VieNeu, thử giọng chọn đoạn mẫu
│   ├── chapter_builder.py  Đọc cả chương, nghe lại + đọc lại chunk lỗi, hậu kỳ, timeline
│   ├── asr_checker.py      PhoWhisper nghe lại theo batch, tỷ lệ sai ký tự (bỏ qua tên riêng), lặp từ
│   ├── audio_utils.py      Đọc/ghi WAV, resample, cắt lặng
│   ├── audio_master.py     Hậu kỳ FFmpeg: lọc ù, chuẩn LUFS 2 lượt, limiter
│   ├── video_maker.py      Đơn vị lặp video nền, nối audio tập, xuất MP4
│   ├── drive.py            Upload rclone
│   └── utils.py            Log, JSON, fingerprint
├── assets/voice/           Giọng mẫu (file .wav không đưa lên GitHub)
├── docs/HUONG_DAN_RUNPOD.md
└── tests/                  Kiểm thử: python -m pytest tests
```

### Dữ liệu trên pod (`/workspace/audio_studio`; mỗi bộ truyện nằm riêng trong `projects/<tên bộ>/` với cùng cấu trúc)

```
/workspace/audio_studio/
├── config.yaml             Cấu hình bạn sửa (ưu tiên hơn bản trong code)
├── rclone.conf             Google Drive (tự ghi từ RunPod Secret RCLONE_CONF_B64)
├── input/                  Chương .txt, glossary.json, names.txt
├── assets/voice/           Giọng mẫu
├── assets/background/      Video nền
├── Kenh/                   Intro, watermark
├── output/audio/           Chuong_0001.wav … (WAV 24-bit 48 kHz đã hậu kỳ)
├── output/video/           Chuong_0001-0020_Tap001.mp4 (hoặc .done sau khi lên Drive)
├── work/
│   ├── voice/              Đoạn giọng mẫu đã chọn (ref_*.wav) + điểm thử giọng (ref_*.json)
│   ├── chXXXX/timeline.json  Số chunk, thời gian đọc/ASR, các chunk còn nghi lỗi (warnings)
│   ├── bench/              Kết quả `bench`: report.json + file nghe thử chunk bị gắn cờ
│   ├── video/              Đơn vị lặp video nền, intro đã chuẩn hóa
│   ├── logs/               Log theo ngày
│   └── studio_run.log      Log chạy nền
└── .cache/huggingface/     Model VieNeu, MOSS codec, PhoWhisper (tải 1 lần)
```

## Tùy chỉnh thường gặp (`config.yaml`)

| Muốn | Sửa |
|---|---|
| To hơn / nhỏ hơn | `master.lufs`: −14 (mặc định, chuẩn YouTube), −13 to hơn, −16 nhỏ hơn |
| Nhanh hơn (rẻ hơn) | `asr.enabled: false` (xem kết quả `bench`), hoặc `asr.model_id: vinai/PhoWhisper-medium` |
| Đọc chậm/nhanh | VieNeu không có tham số tốc độ: nhịp đọc theo giọng mẫu → chọn giọng mẫu đọc chậm hơn |
| Một từ bị đọc sai | thêm vào `chuong/glossary.json` trên Drive: `{"từ gốc": "cách đọc"}` |
| ASR báo sai ở tên nhân vật | thêm tên vào `chuong/names.txt` trên Drive |
| Giọng clone chưa giống | giọng mẫu sạch hơn (không nhạc nền), hoặc xóa `work/voice/` để chọn lại đoạn mẫu |
| Video nền giật ở chỗ lặp | tăng `video.loop_crossfade_sec` |
| Ép độ phân giải | `video.resolution: [1920, 1080]` |
| Số chương mỗi tập, intro/watermark | `episode.chapters_per_episode`, `episode.inserts` |
| Đổi giọng | `voice.ref_audio` (tool tự chọn lại đoạn mẫu; chương đã có audio giữ nguyên, muốn đọc lại thì `--force`) |

Sau khi chạy, mở `work/chXXXX/timeline.json` và xem mục `warnings`: đó là các chunk vẫn nghi lỗi sau mọi lần đọc lại, kèm văn bản và điều ASR nghe được.
