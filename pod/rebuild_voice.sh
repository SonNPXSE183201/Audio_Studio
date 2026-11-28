#!/usr/bin/env bash
# MỘT LỆNH làm lại toàn bộ giọng fine-tune trên pod mới (không cần trông):
#   cài đặt -> tải giọng mẫu + chương thử từ Drive -> tải các đoạn YouTube -> tách nhạc (Demucs)
#   -> dataset -> train LoRA (cấu hình tốt nhất: --target all, 8 epoch) -> chấm điểm
#   -> đẩy lên thư mục Drive <folder_id> -> kiểm tra trên Drive -> (tùy chọn) Stop pod.
#
#   nohup bash pod/rebuild_voice.sh <drive_folder_id> [--stop-pod] > /workspace/rebuild.log 2>&1 &
set -Eeuo pipefail

FOLDER_ID="${1:?id thư mục Drive}"; STOP="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
V="${DATA_DIR}/voice_lora"
STORY="${STORY:-Đại Niết Bàn}"
VOICE="${VOICE_NAME:-Giọng đọc truyện}"
SECTIONS="${SECTIONS:-rusHUC8Levs@01:00:00-01:10:00 ZUHbS4D9QDg@01:00:00-01:10:00 C0XaHIxWsM0@01:00:00-01:10:00 GSC-LJh4jgk@02:00:00-02:10:00 rusHUC8Levs_b@02:00:00-02:10:00}"
say() { echo "[$(date +%H:%M:%S)][REBUILD] $*"; }

say "1/4 - Cài đặt"
bash "${PROJECT_DIR}/pod/setup_pod.sh"
source "${DATA_DIR}/env.sh"
PY="${AUDIO_STUDIO_PYTHON:-python3}"
RC=(--config "${DATA_DIR}/rclone.conf")

say "2/4 - Giọng mẫu + chương thử từ Drive"
mkdir -p "${V}/raw" "${V}/ref"
rclone copy "${RC[@]}" "gdrive:Dự án truyện/${STORY}/voice" "${V}/ref" --include 'voice_reference_clean.wav'
rclone copy "${RC[@]}" "gdrive:Dự án truyện/${STORY}/chuong" "${V}/ref" --include 'Chuong_0100.txt'
[[ -s "${V}/ref/voice_reference_clean.wav" && -s "${V}/ref/Chuong_0100.txt" ]] || { say "Thiếu giọng mẫu/chương thử"; exit 1; }

say "3/4 - Tải YouTube + tách nhạc + dataset"
"${PY}" -m pip install -q yt-dlp demucs
cd "${V}/raw"
for s in ${SECTIONS}; do
  out="${s%@*}"; id="${out%_b}"
  [[ -s "${out}.wav" ]] && continue
  for try in 1 2 3; do
    say "Tải ${out} (${s#*@}) lần ${try}"
    rm -f "${out}".*part
    "${PY}" -m yt_dlp -q --no-warnings -f bestaudio -x --audio-format wav \
      --postprocessor-args "ffmpeg:-ac 1 -ar 44100" --download-sections "*${s#*@}" \
      -o "${out}.%(ext)s" "https://www.youtube.com/watch?v=${id}" && break
    sleep 5
  done
  [[ -s "${out}.wav" ]] || { say "Không tải được ${out}"; exit 1; }
done
cd "${V}"
"${PY}" -m demucs -n htdemucs --two-stems vocals -d cuda -o demucs raw/*.wav
cd "${PROJECT_DIR}"
"${PY}" finetune/build_voice_dataset.py --demucs "${V}/demucs/htdemucs" --ref "${V}/ref/voice_reference_clean.wav" \
    --out "${V}/dataset" --target-minutes 40

say "4/4 - Train + chấm điểm + Drive"
EPOCHS="${EPOCHS:-8}" LORA_ARGS="${LORA_ARGS:---target all}" RUN_TAG="${RUN_TAG:-_all}" \
  bash "${PROJECT_DIR}/pod/voice_pipeline.sh" "${V}/dataset" "${VOICE}" "${V}/ref/voice_reference_clean.wav" \
    "${V}/ref/Chuong_0100.txt" "${FOLDER_ID}" ${STOP}
say "XONG"
