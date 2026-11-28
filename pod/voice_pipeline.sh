#!/usr/bin/env bash
# Chạy không cần trông: đợi dataset -> train LoRA -> chấm điểm -> đẩy lên Drive -> (tùy chọn) Stop pod.
#
#   nohup bash pod/voice_pipeline.sh <dataset> "<Tên giọng>" <giọng mẫu.wav> <chương thử.txt> \
#         <drive_folder_id> [--stop-pod] > voice_pipeline.log 2>&1 &
#   EPOCHS="3 5 8 12" (biến môi trường) đổi các mức epoch train; mặc định "2 3 5".
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
# --timeout: một lần upload Drive bị treo (đã gặp: file model 236 MB đứng ~1 KB/s hơn 20 phút) thì
# rclone bỏ kết nối đó và thử lại thay vì chờ mãi; chunk 64M để file lớn lên nhanh.
RC=(--config "${DATA_DIR}/rclone.conf" --drive-root-folder-id "${FOLDER_ID}" --timeout 2m --contimeout 30s
    --retries 5 --low-level-retries 20 --drive-chunk-size 64M)
# Đích PHẢI có tiền tố remote: thiếu "gdrive:" thì rclone chép vào thư mục CỤC BỘ trên pod, rclone check
# so cục bộ cũng "khớp" -> tưởng đã lên Drive rồi Stop pod -> mất hết (đã xảy ra 28/11/2026).
DST="${DRIVE_REMOTE:-gdrive}:"
say() { echo "[$(date +%H:%M:%S)][PIPELINE] $*"; }
rclone lsf "${RC[@]}" "${DST}" --max-depth 1 >/dev/null || { say "Không truy cập được thư mục Drive ${FOLDER_ID}"; exit 1; }

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

VOICE_CLIP="${CLIP}" bash "${PROJECT_DIR}/pod/train_voice.sh" "${DATASET}" "${VOICE}" "${REF}" "${CHAPTER}" ${EPOCHS:-2 3 5}
BEST="$(cat "${WORK}/eval/goi_y.txt" 2>/dev/null || true)"
say "Model gợi ý: ${BEST:-không model fine-tune nào hơn clone tức thì}"
# Điểm giống giọng so với chính người đọc trong dataset (kèm trần giọng thật). Lỗi thì bỏ qua.
( cd "${PROJECT_DIR}" && "${PY}" finetune/score_voice.py --dataset "${DATASET}" --ref "${REF}" \
    "${WORK}"/eval/*.wav 2>&1 | grep -v -i -E 'warn' > "${WORK}/eval/diem_giong.txt" ) || true
cat "${WORK}/eval/diem_giong.txt" 2>/dev/null || true

say "Đẩy kết quả lên Drive (thư mục ${FOLDER_ID})..."
cp "${CLIP}" "${WORK}/eval/clip_dong_goi_giong.wav"
cp "${WORK}"/*.log "${WORK}/eval/" 2>/dev/null || true
rclone copy "${RC[@]}" "${WORK}/eval" "${DST}eval"
rclone copy "${RC[@]}" "${DATASET}" "${DST}dataset" --exclude "train.parquet" --exclude "_clips/**"
for run in "${WORK}"/output/*/; do
  name="$(basename "${run}")"
  rclone copy "${RC[@]}" "${run}adapter" "${DST}models/${name}/adapter"
  rclone copy "${RC[@]}" "${run}merged" "${DST}models/${name}/merged"
done

say "Kiểm tra Drive đủ file..."
ok=1
rclone check "${RC[@]}" "${WORK}/eval" "${DST}eval" --one-way || ok=0
for run in "${WORK}"/output/*/; do
  name="$(basename "${run}")"
  rclone check "${RC[@]}" "${run}merged" "${DST}models/${name}/merged" --one-way || ok=0
done
# Xác nhận bằng một lần liệt kê THẬT trên Drive (không tin riêng mã thoát của copy/check).
remote_n="$(rclone lsf "${RC[@]}" "${DST}models" -R --files-only 2>/dev/null | wc -l)"
[[ "${remote_n}" -gt 0 ]] || ok=0
if [[ "${ok}" != 1 ]]; then
  say "Drive CHƯA đủ file (${remote_n} file trong models/) -> KHÔNG tắt pod (tránh mất model). Chạy lại phần rclone copy."
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
