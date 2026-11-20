# Hướng dẫn chạy Audio Studio trên RunPod

Mục tiêu: đọc hàng trăm chương thành video với chi phí pod thấp nhất, giọng đọc vẫn mượt và rõ.

## 0. Tóm tắt nhanh

**Cách khuyên dùng: mọi thứ để trên Google Drive, pod mới chỉ cần 1 lệnh.**

Chuẩn bị một lần (đã làm xong; chi tiết ở mục 3 và 5):
1. Drive: `Dự án truyện/<tên bộ>/` gồm `chuong/`, `voice/`, `background/`, `Kenh/`, `video/` (copy từ `Dự án truyện/Sample`).
2. RunPod **Secret** `rclone_conf` = nội dung `rclone.conf` mã hóa base64.
3. RunPod **Template** `Audio Studio` (image PyTorch CUDA 12.8, ổ 40 GB, biến `RCLONE_CONF_B64`).

Mỗi lần chạy: **Pods → Deploy** → chọn GPU → template **Audio Studio** → Deploy. Pod lên thì **Connect → Jupyter (8888) → Terminal** và chạy (tên bộ = tên thư mục trên Drive):

```bash
curl -fsSL https://raw.githubusercontent.com/SonNPXSE183201/Audio_Studio/main/pod/bootstrap.sh \
  | AUTO_RUN=1 PROJECTS="Đại Niết Bàn" JOBS=2 bash
```

Nhiều bộ truyện: ngăn cách bằng dấu `;`, ví dụ `PROJECTS="Đại Niết Bàn;Trọng sinh 2003: Giang Thần"`: chạy lần lượt từng bộ, bộ này xong mới sang bộ sau. Tên phải **giống hệt** tên thư mục trong `Dự án truyện` trên Drive (cả dấu, dấu cách).

Thêm bộ mới: trên Drive copy thư mục `Dự án truyện/Sample`, đổi tên thành tên bộ, thả chương vào `chuong/`, giọng mẫu vào `voice/`, video nền vào `background/`, intro/watermark vào `Kenh/`. Video tự đổ về `video/`.

Tool tự cài đặt, tự tải chương và tài nguyên từ Drive, bỏ qua các tập đã có trên Drive, đọc audio, ghép tập, upload. Xem tiến độ: `bash /root/Audio_Studio/run.sh log`.

Thêm chương mới lên Drive trong lúc pod đang chạy: tool tự lấy về sau tối đa 20 phút.

**Tự tắt pod khi xong** (giống AI_Video_Studio): thêm `STOP_POD=1` vào lệnh trên. Khi mọi bộ đã đọc xong, tool ghép nốt tập lẻ cuối, đẩy hết video lên Drive rồi mới **Stop** pod. Còn video nào chưa lên Drive thì **không** tắt (tránh mất video). `STOP_POD=terminate` thì xóa hẳn pod.

```bash
curl -fsSL https://raw.githubusercontent.com/SonNPXSE183201/Audio_Studio/main/pod/bootstrap.sh   | AUTO_RUN=1 PROJECTS="Đại Niết Bàn" JOBS=2 STOP_POD=1 bash
```

Pod đang chạy sẵn: thêm `--stop-pod` vào lệnh start (job đang chạy thì chỉ bật thêm chế độ tự tắt):

```bash
bash /root/Audio_Studio/run.sh start --project "Đại Niết Bàn" --jobs 2 --stop-pod
bash /root/Audio_Studio/run.sh status --project "Đại Niết Bàn"   # dòng "Tự tắt pod: BẬT"
tail -f "/workspace/audio_studio/projects/Đại Niết Bàn/work/logs/stop_pod.log"
```

Chế độ tự tắt pod chỉ tắt khi **thật sự xong**:
- không còn job nào chạy, kiểm tra 2 lần liên tiếp;
- hỏi tool còn chương nào chưa đọc: còn (job chết vì hết VRAM, lỗi mạng…) thì **tự chạy lại job** (tối đa 3 lần) thay vì tắt pod;
- ghép tập lẻ cuối, đẩy nốt video; còn video chưa lên Drive thì **không** tắt.

`bash run.sh stop` dừng job **và tắt luôn chế độ tự tắt pod**, nên dừng/sửa/chạy lại thoải mái không lo pod bị tắt nhầm. Chạy lại thì nhớ thêm `--stop-pod`.

## 1. Tài khoản và lưu trữ

### Mặc định: không cần Network Volume (template Audio Studio)

Chương lấy từ Drive, video đẩy về Drive, nên chỉ cần **ổ container 40 GB** (model ~6 GB, audio 20 chương, video tập đang chờ upload).
- Mỗi lần tạo pod mới phải cài lại và tải lại model: **~10 phút**.
- Stop/Terminate pod là mất những gì **chưa lên Drive** (tối đa ~20 chương đang làm dở): nên thuê **On-Demand**, không thuê Spot.
- Pod mới tự biết tập nào đã xong nhờ xem thư mục video trên Drive (`sync.skip_uploaded: true`), không cần nhớ `--range`.
- `upload.delete_audio_after_upload: true` (mặc định) giữ ổ container không đầy.

**Chi phí bộ 1.500 chương (~200 giờ audio, 75 tập × 20 chương), không volume:**

| GPU (giá/giờ) | 1 job | 2 job / 1 pod | 2 pod × 1 job |
|---|---|---|---|
| RTX A5000 ($0.16–0.27) | ~50 giờ, **$8–14** | ~42–46 giờ, **$7–12** | ~25 giờ, $8–14 |
| RTX 5090 ($0.69–0.99) | ~18 giờ, $13–18 | ~15–17 giờ, $11–16 | **~9 giờ**, $13–18 |
| RTX 4090 ($0.89) | ~22 giờ, $20 | ~19–20 giờ, $17 | ~11 giờ, $20 |
| RTX PRO 4500 SE ($0.72) | ~31 giờ, $22 | ~26–28 giờ, $19 | ~15 giờ, $22 |
| A40 ($0.59) | ~44 giờ, $26 | ~37–40 giờ, $23 | ~22 giờ, $26 |

Cộng thêm ổ container (~$0.1–0.3 cả dự án) và ~10 phút mỗi lần tạo pod mới. Video 75 tập (20 chương/tập) ≈ 300–650 GB trên Drive.

## 2. Chọn GPU: thời gian và chi phí theo cấu hình

OmniVoice (đọc) và PhoWhisper (nghe lại) là model nhỏ, chỉ dùng vài GB VRAM. Mọi GPU 24 GB trở lên đều chạy được với chất lượng **như nhau**. Khác biệt chỉ là **tốc độ**, mà tốc độ quyết định **chi phí**.

Quy ước: 1 chương ≈ **8 phút audio** (khoảng 11.000 ký tự), 1 tập = 20 chương ≈ 160 phút. Cấu hình mặc định: `num_step: 32`, ASR bật, đọc lại tối đa 3 lượt.

| GPU | VRAM | Giá/giờ | Batch tự chọn | Tốc độ (× thời gian thực) | 1 chương | 1 tập (20 chương) | Chương / 1 giờ pod | Chi phí 1 tập | Chi phí 100 chương |
|---|---|---|---|---|---|---|---|---|---|
| RTX A5000 | 24 GB | $0.16–0.27 ¹ | 8 | ~3–5× | ~2 phút | ~40 phút | ~30 | **$0.10–0.18** | **$0.5–0.9** |
| RTX 3090 | 24 GB | $0.22–0.46 ¹ | 8 | ~3,5–5,5× | ~1,8 phút | ~36 phút | ~34 | $0.14–0.28 | $0.7–1.4 |
| L4 | 24 GB | $0.43–0.59 ¹ | 8 | ~2–3,5× | ~2,9 phút | ~58 phút | ~21 | $0.42–0.56 | $2.0–2.8 ❌ |
| A40 | 48 GB | $0.59 ² | 16 | ~3,5–5,5× | ~1,8 phút | ~36 phút | ~34 | $0.36 | $1.75 |
| RTX PRO 4500 SE | 32 GB | $0.72 ² | 8 | ~5–8× ³ | ~1,2 phút | ~24 phút | ~49 | $0.30 | $1.5 |
| RTX 4090 | 24 GB | $0.89 ² | 8 | ~7–11× | ~0,9 phút | ~18 phút | ~67 | $0.26 | $1.3 |
| RTX 5090 | 32 GB | $0.69–0.99 ¹ | 8 | ~9–14× | ~0,7 phút | ~14 phút | ~82 | **$0.16–0.24** | **$0.8–1.2** |
| L40S | 48 GB | $1.09 ² | 16 | ~7–10× | ~0,9 phút | ~19 phút | ~64 | $0.34 | $1.7 |
| RTX PRO 6000 | 96 GB | $2.69 ² | 24 | ~9–15× | ~0,7 phút | ~14 phút | ~90 | $0.60 | $3.0 ❌ |
| A100 / H100 | 80 GB | $1.2–2.7 | 24 | ~8–15× | — | — | — | đắt | ❌ không đáng |

¹ Giá công bố trên runpod.io (Community–Secure Cloud, 2026). Giá thật trên tài khoản có thể khác, xem trang Deploy.
² Giá trên tài khoản của bạn ngày 20/11/2026 (theo hướng dẫn AI_Video_Studio).
³ **Mốc đo thật**: trên RTX PRO 4500, cách đọc cũ (từng câu một, mã hóa lại giọng mẫu mỗi câu) đọc 7,5 phút audio trong 3,8 phút, tức **~2× thời gian thực**. Bản này sinh theo batch, mã hóa giọng mẫu 1 lần và nghe lại theo batch, nên ước tính nhanh hơn 2,5–4 lần. Các GPU khác được suy ra theo sức tính FP16 tương đối. **Đây là ước tính, chưa đo trên từng pod.** Đo thật bằng lệnh `bench` bên dưới.

**Nên thuê:**
1. **RTX 5090** hoặc **RTX A5000 / 3090 (Community)**: rẻ nhất trên mỗi chương. 5090 nhanh nhất trong nhóm rẻ.
2. **RTX 4090 / RTX PRO 4500 SE**: khi nhóm trên hết máy.
3. Không nên thuê L4 (băng thông bộ nhớ thấp, chậm), RTX PRO 6000, A100, H100 (model nhỏ, không tận dụng được GPU lớn: trả gấp 3–4 lần tiền mà chỉ nhanh hơn chút ít).

**Đo thật trước khi chạy cả truyện (5 phút, vài cent):**

```bash
bash run.sh bench --price 0.69          # thay 0.69 bằng giá pod đang thuê
```

Kết quả mẫu:

```
[Bench] Máy: NVIDIA GeForce RTX 5090 (VRAM dùng tối đa 6.2 GB)
[Bench] Đọc 48 câu = 6.9 phút audio trong 0.6 phút -> nhanh gấp 11.5 lần thời gian thực (2 câu phải đọc lại)
[Bench] 1 chương ~8 phút audio: ~0.7 phút GPU
[Bench] 1 tập 20 chương: ~14 phút
[Bench] 1 giờ pod: ~86 chương (~11.5 giờ audio)
[Bench] Giá $0.69/giờ: ~$0.008/chương, ~$0.16/tập, ~$0.060/giờ audio
```

(Số trong khối trên chỉ minh họa định dạng, không phải số đo.) Có thể thuê 2–3 loại GPU, chạy `bench` trên mỗi loại rồi giữ lại loại có **"$/tập" thấp nhất**.

**Lưu ý khi thuê:**
- Cần **≥ 4 vCPU** và **≥ 16 GB RAM**: ghép video, hậu kỳ audio và ASR dùng CPU. Pod ít CPU thì GPU lớn cũng phải chờ.
- GPU Blackwell (RTX 5090, RTX PRO 4500/6000) cần template **CUDA 12.8**.
- Nên dùng **Community Cloud** (rẻ hơn). Có thể thuê **Spot/Interruptible**: bị thu hồi giữa chừng cũng không mất việc, vì tool lưu cache sau mỗi 64 câu và chạy lại thì đọc tiếp.
- Mặc định `tts.batch_size: 1`: mỗi câu sinh riêng như AI_Video_Studio (giọng đồng đều nhất). Một job chỉ dùng ~5 GB VRAM, nên để tận dụng GPU hãy chạy **`JOBS=2`–`3`** (24 GB) thay vì tăng batch. `batch_size: auto` nhanh hơn nhưng mỗi câu một kiểu ngữ điệu.

## 3. Deploy pod

Dùng template **Audio Studio** đã tạo sẵn (RunPod → **Templates**). Cấu hình của template, nếu cần tạo lại:

| Mục | Giá trị |
|---|---|
| Template name | `Audio Studio` (Type: Pod, Compute: NVIDIA GPU, Private) |
| Container image | `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404` (CUDA 12.8: chạy cả RTX 5090 / PRO 4500) |
| Container disk | **40 GB** (model ~6 GB + audio 20 chương + video 1 tập ~8 GB) |
| Volume | 0 GB: không cần, mọi thứ lấy từ / đẩy về Google Drive |
| HTTP port | `Jupyter` 8888 (mở Terminal, kéo-thả file) |
| TCP port | `SSH` 22 |
| Environment variable | `RCLONE_CONF_B64` = `{{ RUNPOD_SECRET_rclone_conf }}` (bấm biểu tượng chìa khóa ở ô value để chọn secret) |

Khi Deploy: chọn **On-Demand** (Spot bị thu hồi thì mất phần chưa lên Drive, tối đa ~20 chương), GPU theo bảng ở mục 2, ≥ 4 vCPU và ≥ 16 GB RAM.

## 4. Cài đặt (chạy lại sau mỗi lần Start pod)

Mở Terminal của pod (Web Terminal hoặc Jupyter → Terminal):

```bash
curl -fsSL https://raw.githubusercontent.com/SonNPXSE183201/Audio_Studio/main/pod/bootstrap.sh | bash
```

Lệnh này:
1. clone hoặc cập nhật code vào `/root/Audio_Studio`;
2. cài `ffmpeg`, `rclone`, `omnivoice`, `transformers` vào venv dùng chung PyTorch CUDA của template;
3. tạo cấu trúc `/workspace/audio_studio`, chép `config.yaml` mặc định (nếu chưa có);
4. tải trước OmniVoice và PhoWhisper lên volume (chỉ lần đầu, các lần sau bỏ qua).

Log cài đặt nằm ở `/workspace/audio_studio_setup.log`.

## 5. Dữ liệu: tự lấy từ Google Drive (khuyên dùng)

### 5.1. Sắp xếp thư mục trên Drive

```
Drive của tôi/Dự án truyện/
├── Sample/            thư mục mẫu (chuong, voice, background, Kenh, video): copy + đổi tên cho bộ mới
└── Đại Niết Bàn/      tên thư mục = tên bộ truyện trong PROJECTS
    ├── chuong/        Chuong_0001.txt ... Chuong_1500.txt (+ glossary.json, names.txt nếu có)
    ├── voice/         voice_reference_clean.wav + voice_reference_clean.txt
    ├── background/    background.mp4
    ├── Kenh/          Script_Intro.mp3, Script_Watermark.mp3, Script_Watermark1.mp3
    └── video/         tool upload video tập vào đây
```

Tên bộ có dấu, dấu cách, dấu `:` đều được, miễn **giống hệt** tên thư mục trên Drive.

### 5.2. `config.yaml`: không cần sửa

`config.yaml` trong code đã trỏ đúng cấu trúc trên; `{project}` được thay bằng tên bộ khi chạy:

```yaml
sync:
  enabled: auto        # tự bật khi pod có rclone.conf
  input: "gdrive:Dự án truyện/{project}/chuong"
  assets:
    assets/voice: "gdrive:Dự án truyện/{project}/voice"
    assets/background: "gdrive:Dự án truyện/{project}/background"
    Kenh: "gdrive:Dự án truyện/{project}/Kenh"
upload:
  enabled: auto
  remote_dir: "gdrive:Dự án truyện/{project}/video"
```

Bộ nào cần cấu hình khác (vd. giữ 10 chương/tập) thì tạo `/workspace/audio_studio/projects/<tên bộ>/config.yaml`, chỉ ghi những dòng muốn đổi.

### 5.3. `rclone.conf` qua RunPod Secret (làm một lần, pod nào cũng tự có)

`rclone.conf` là "chìa khóa" vào Google Drive, nên **pod không thể tự lấy nó từ Drive**. Thay vào đó, lưu nó trong RunPod Secret:

1. Trên máy Windows, mở PowerShell, chép chuỗi base64 vào clipboard:
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("D:\Dự án truyện\Tool\rclone.conf")) | Set-Clipboard
   ```
2. RunPod → **Secrets** → **Create secret**: tên `rclone_conf`, giá trị: Ctrl+V.
3. Template **Audio Studio** đã có biến `RCLONE_CONF_B64` = `{{ RUNPOD_SECRET_rclone_conf }}` (mục 3).

Kiểm tra trên pod: log cài đặt có dòng `[OK] rclone.conf từ biến RCLONE_CONF_B64`. Nếu báo lỗi base64, làm lại bước 1 rồi dán đè vào secret. Đổi tài khoản Drive hoặc token bị thu hồi thì cũng chỉ cần cập nhật secret.

(Tùy chọn) Muốn thay cả `config.yaml` mà không sửa code: tạo secret `studio_config` từ file config mã hóa base64, thêm biến `STUDIO_CONFIG_B64` = `{{ RUNPOD_SECRET_studio_config }}` vào template.

### 5.4. Hoặc chép tay lên pod

Cần có:

```
/workspace/audio_studio/
├── input/Chuong_0001.txt ...
├── input/names.txt                       (tùy chọn) tên nhân vật, mỗi dòng 1 tên
├── input/glossary.json                   (tùy chọn) cách đọc riêng
├── assets/voice/voice_reference_clean.wav
├── assets/voice/voice_reference_clean.txt
├── assets/background/background.mp4
├── Kenh/Script_Intro.mp3, Script_Watermark.mp3, Script_Watermark1.mp3
└── rclone.conf
```

Cách chép:
- **Jupyter**: kéo-thả file vào đúng thư mục (file zip thì giải nén bằng `unzip -o file.zip -d /workspace/audio_studio/input`).
- **Từ Google Drive** (sau khi có `rclone.conf`):
  ```bash
  rclone copy "gdrive:Dự án truyện/Ten_truyen/chuong" /workspace/audio_studio/input --config /workspace/audio_studio/rclone.conf -P
  ```
- **Từ máy tính**: `runpodctl send file.zip` trên máy, rồi `runpodctl receive <mã>` trên pod.

> Giọng mẫu **không có trong repo GitHub** (repo công khai). Hãy tự chép file `.wav` và `.txt` vào `assets/voice/`.

## 6. Google Drive

1. Trên máy tính: `rclone config` → tạo remote tên `gdrive` (Google Drive).
2. Chép file `rclone.conf` (Windows: `%APPDATA%\rclone\rclone.conf`) vào `/workspace/audio_studio/rclone.conf`.
3. Sửa `/workspace/audio_studio/config.yaml`:
   ```yaml
   upload:
     enabled: true
     remote_dir: "gdrive:Dự án truyện/Ten_truyen"
   ```

Upload thành công thì video bị xóa khỏi pod và để lại file `.done`, nên lần sau tool không ghép lại. Upload lỗi thì giữ video, lần chạy sau thử lại.

## 7. Chạy

```bash
cd /root/Audio_Studio
bash run.sh check                 # mọi dòng phải OK, video nền và giọng mẫu không được THIẾU
bash run.sh start                 # chạy nền, đóng trình duyệt vẫn chạy
bash run.sh log                   # xem tiến độ
```

Khi chạy, GPU đọc audio từng chương. Đủ 20 chương thì CPU ghép tập và upload **trong lúc GPU đọc tiếp**.

Hết truyện (tập cuối chưa đủ 20 chương):

```bash
bash run.sh video --finish
```

### Nhiều bộ truyện, nhiều job

```bash
bash run.sh start --project "Đại Niết Bàn" --jobs 2   # 2 job audio tự chia chương + 1 job video chạy ngầm
bash run.sh status --project "Đại Niết Bàn"
bash run.sh log --project "Đại Niết Bàn"
bash run.sh stop --project "Đại Niết Bàn"
```

- Mỗi bộ truyện có dữ liệu riêng ở `/workspace/audio_studio/projects/<tên>/`, lấy chương từ `gdrive:Dự án truyện/<tên>/chuong` (theo `{project}` trong `config.yaml`).
- Các job audio **tự chia chương** bằng khóa: job nào rảnh thì lấy chương kế tiếp chưa ai làm, không đọc trùng, không cần `--range`.
- Job video chạy bằng CPU, đủ 20 chương liên tiếp là ghép tập và upload, mỗi tập chỉ ghép một lần, tự dừng khi audio xong hết.
- `--jobs 2` có nhanh hơn 1 job không thì tùy GPU: xem `nvidia-smi`, GPU-Util dưới ~70% thì mới đáng chạy 2 job (mục 2). Mỗi job tự giảm batch còn một nửa để vừa VRAM.

**2 pod cho một bộ truyện:** khóa chương chỉ có tác dụng trong cùng một pod, nên 2 pod phải chia chương bằng `--range` theo bội số 20:

```bash
bash run.sh start --project "Đại Niết Bàn" --range 1-750      # pod 1
bash run.sh start --project "Đại Niết Bàn" --range 751-1500   # pod 2
```

## 8. Xem tool đang làm gì

| Muốn xem | Lệnh / nơi xem |
|---|---|
| Tiến độ | `bash run.sh log` |
| Tốc độ mỗi chương | dòng `Chương N: ... audio trong ... (xK thời gian thực)` |
| GPU có chạy đầy không | `nvidia-smi -l 2` |
| Câu còn nghi lỗi | `work/chXXXX/timeline.json` → `warnings` (có `time_sec` để nghe kiểm tra) |
| Audio chương | `output/audio/Chuong_XXXX.wav` (tải về nghe qua Jupyter) |
| Log upload | `work/upload.log` |

## 9. Tắt pod rồi bật lại

1. `bash run.sh stop` (hoặc cứ để, Stop pod cũng được: cache câu không mất).
2. Stop pod trên RunPod. Volume vẫn tính tiền lưu trữ nhưng rất rẻ.
3. Start lại, chạy lại lệnh `bootstrap.sh` ở mục 4, rồi `bash run.sh start`: tool đọc tiếp đúng chỗ dừng.

## 10. Chi phí và tối ưu

**Chi phí cố định:**

| Khoản | Chi phí |
|---|---|
| Network Volume 50 GB | ~$3.5/tháng (~$0.07/GB/tháng), tính cả khi pod tắt |
| Ổ container 30 GB | vài cent/ngày, chỉ tính khi pod chạy |
| Mỗi lần Start pod | 3–5 phút cài lại gói Python (model đã nằm trên volume). Lần đầu thêm 5–10 phút tải model |
| Ghép video 1 tập | 1–3 phút CPU, chạy song song với GPU nên **không tốn thêm tiền** |

**Ví dụ cả bộ truyện 800 chương (80 tập, ~107 giờ audio), theo bảng ở mục 2:**

| GPU | Thời gian chạy | Chi phí pod |
|---|---|---|
| RTX A5000 Community ($0.16) | ~27 giờ | **~$4–7** |
| RTX 5090 ($0.69) | ~10 giờ | **~$7–10** |
| RTX 4090 ($0.89) | ~12 giờ | ~$11 |
| RTX PRO 4500 SE ($0.72) | ~16 giờ | ~$12 |
| A40 ($0.59) | ~24 giờ | ~$14 |

So với cách đọc cũ (~2× thời gian thực, ~53 giờ trên RTX PRO 4500 ≈ $38 cho 800 chương), chi phí ước tính giảm còn khoảng 1/3 đến 1/8.

**Tác động của cấu hình lên tốc độ (ước tính):**

| Thay đổi | Nhanh hơn | Ảnh hưởng chất lượng |
|---|---|---|
| `asr.model_id: vinai/PhoWhisper-medium` | ~10% | gần như không |
| `tts.num_step: 24` | ~20–25% | nhẹ, nghe thử trước |
| `asr.enabled: false` | ~15–25% | **không còn tự sửa câu vấp** (vẫn bắt câu câm, kẹt, cụt đuôi) |
| `asr.max_attempts: 2` | ít (chỉ câu lỗi) | câu lỗi được ít cơ hội sửa hơn |
| Giọng mẫu ngắn hơn (8–10 s thay vì 18,8 s) | ~10–20% | phải cắt đúng câu và sửa `.txt` cho khớp |

**Mẹo tiết kiệm:**
- **Đừng để pod chạy không**: chạy hết chương rồi Stop pod ngay.
- Thêm nhiều chương một lần (vài trăm chương) để không mất thời gian cài đặt lặp lại.
- Muốn nhanh hơn nữa (theo thứ tự ảnh hưởng chất lượng từ ít đến nhiều):
  1. `asr.model_id: "vinai/PhoWhisper-medium"`: nghe lại nhanh hơn, vẫn bắt được lỗi vấp;
  2. `tts.num_step: 24`: nhanh hơn khoảng 25%, nghe thử 1 chương trước khi áp dụng cả truyện;
  3. `asr.enabled: false`: bỏ nghe lại (vẫn giữ kiểm tra tín hiệu: câu câm, kẹt, cụt đuôi).
- Video nền chỉ được nén **1 lần** (đơn vị lặp, cache trong `work/video/`). Ghép mỗi tập chỉ nén audio, nên mất vài phút CPU, không tốn GPU.

## 11. Sự cố thường gặp

| Dấu hiệu | Cách xử lý |
|---|---|
| `PyTorch không thấy GPU` | chọn lại template RunPod PyTorch (CUDA) |
| `Batch ... hết VRAM, chia nhỏ` liên tục | đặt `tts.batch_size` nhỏ hơn hoặc `batch_max_chars: 1200` |
| Giọng đầu câu bị lạ / vấp | lời trong `voice_reference_clean.txt` không khớp giọng mẫu; sửa cho khớp từng chữ |
| Một từ luôn đọc sai | thêm vào `chuong/glossary.json` trên Drive |
| Nhiều câu báo `sai xx% ký tự` dù nghe vẫn đúng | thường do tên riêng: thêm vào `chuong/names.txt` trên Drive |
| `Không sinh được audio cho câu ...` (dừng chương) | câu có ký tự lạ; sửa câu đó trong file `.txt` rồi chạy lại (các câu khác đã có cache) |
| Video ra ngắn hơn audio | xem log `Video ra ...s nhưng audio ...s`; video nền hỏng, hãy thử file nền khác |
| Upload lỗi | kiểm tra `rclone.conf`, `upload.remote_dir`; xem `work/upload.log` |
| Muốn đọc lại 1 chương | `bash run.sh audio --range 15-15 --force` rồi xóa video tập chứa chương đó và chạy `bash run.sh video` |
