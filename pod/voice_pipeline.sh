#!/usr/bin/env bash
# Chạy không cần trông: đợi dataset -> train LoRA -> chấm điểm -> đẩy lên Drive -> (tùy chọn) Stop pod.
#
#   nohup bash pod/voice_pipeline.sh <dataset> "<Tên giọng>" <giọng mẫu.wav> <chương thử.txt> \
#         <drive_folder_id> [--stop-pod] > voice_pipeline.log 2>&1 &
#
# Pod KHÔNG có volume thì Stop là mất /workspace: chỉ Stop khi `rclone check` xác nhận Drive đã đủ file.
set -Eeuo pipefail

DATASET="${1:?dataset}"; VOICE="${2:?tên giọng}"; REF="${3:?giọng mẫu}"; CHAPTER="${4:?chương thử}"
FOLDER_ID="${5:?id thư mục Drive}"; STOP="${6:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
[[ -f "${DATA_DIR}/env.sh" ]] && source "${DATA_DIR}/env.sh"
PY="${AUDIO_STUDIO_PYTHON:-python3}"
WORK="${DATA_DIR}/voice_lora"
RC=(--config "${DATA_DIR}/rclone.conf" --drive-root-folder-id "${FOLDER_ID}")
say() { echo "[$(date +%H:%M:%S)][PIPELINE] $*"; }

say "Đợi dataset ${DATASET}/metadata.csv..."
until [[ -s "${DATASET}/metadata.csv" ]]; do sleep 30; done
say "Dataset: $(wc -l < "${DATASET}/metadata.csv") clip"

# Clip đóng gói giọng: clip 5-8 s giống giọng mẫu nhất và sạch nhất trong dataset (44,1 kHz).
CLIP="$("${PY}" - "${DATASET}" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(os.path.join(sys.argv[1], "review.csv"), encoding="utf-8-sig")))
ok = [r for r in rows if 5.0 <= float(r["giay"]) <= 8.0] or rows
best = max(ok, key=lambda r: float(r["giong_giong_mau"]) + 0.1 * float(r.get("dnsmos_sig") or 0))
print(os.path.join(sys.argv[1], "raw_audio", best["file_name"]))
PY
)"
say "Clip đóng gói giọng: ${CLIP}"

VOICE_CLIP="${CLIP}" bash "${PROJECT_DIR}/pod/train_voice.sh" "${DATASET}" "${VOICE}" "${REF}" "${CHAPTER}" 2 3 5
BEST="$(cat "${WORK}/eval/goi_y.txt" 2>/dev/null || true)"
say "Model gợi ý: ${BEST:-không model fine-tune nào hơn clone tức thì}"

say "Đẩy kết quả lên Drive (thư mục ${FOLDER_ID})..."
cp "${CLIP}" "${WORK}/eval/clip_dong_goi_giong.wav"
cp "${WORK}"/*.log "${WORK}/eval/" 2>/dev/null || true
rclone copy "${RC[@]}" "${WORK}/eval" "eval"
rclone copy "${RC[@]}" "${DATASET}" "dataset" --exclude "train.parquet"
for run in "${WORK}"/output/*/; do
  name="$(basename "${run}")"
  rclone copy "${RC[@]}" "${run}adapter" "models/${name}/adapter"
  rclone copy "${RC[@]}" "${run}merged" "models/${name}/merged"
done

say "Kiểm tra Drive đủ file..."
ok=1
rclone check "${RC[@]}" "${WORK}/eval" "eval" --one-way || ok=0
for run in "${WORK}"/output/*/; do
  name="$(basename "${run}")"
  rclone check "${RC[@]}" "${run}merged" "models/${name}/merged" --one-way || ok=0
done
if [[ "${ok}" != 1 ]]; then
  say "Drive CHƯA đủ file -> KHÔNG tắt pod (tránh mất model). Chạy lại phần rclone copy."
  exit 1
fi
say "Drive đã đủ: eval/, dataset/, models/<run>/{adapter,merged}"

if [[ "${STOP}" == "--stop-pod" && -n "${RUNPOD_POD_ID:-}" ]]; then
  say "Stop pod ${RUNPOD_POD_ID}..."
  if [[ -n "${RUNPOD_API_KEY:-}" ]] && curl -fsS -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
      | grep -vq '"errors"'; then
    say "Đã gửi lệnh Stop qua RunPod API."
  elif command -v runpodctl >/dev/null && runpodctl stop pod "${RUNPOD_POD_ID}"; then
    say "Đã gửi lệnh Stop qua runpodctl."
  else
    say "Không tự Stop được: hãy bấm Stop trên RunPod."
  fi
fi
