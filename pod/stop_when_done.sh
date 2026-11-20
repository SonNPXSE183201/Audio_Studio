#!/usr/bin/env bash
# Tự TẮT POD khi bộ truyện đã đọc xong, ghép xong và đẩy hết video lên Google Drive.
#
#   nohup bash /root/Audio_Studio/pod/stop_when_done.sh --project "Đại Niết Bàn" > /workspace/stop_pod.log 2>&1 &
#   bash pod/stop_when_done.sh --project "A" --project "B"     # chờ nhiều bộ
#   bash pod/stop_when_done.sh --project "A" --terminate       # XÓA hẳn pod thay vì Stop
#   bash pod/stop_when_done.sh --project "A" --dry-run         # chỉ kiểm tra, không tắt
#
# Các bước: chờ hết job audio/video -> ghép tập lẻ cuối (--finish) + đẩy lại video lỗi mạng
# -> chỉ tắt khi không còn video .mp4 nào chưa lên Drive (còn thì KHÔNG tắt để khỏi mất video).
# Cách tắt giống AI_Video_Studio: RunPod API (khóa RUNPOD_API_KEY có sẵn trong pod), dự phòng runpodctl.
set -Euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
PROJECTS=()
ACTION="stop"
DRY=0
CHECK_EVERY=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECTS+=("$2"); shift 2 ;;
    --terminate) ACTION="terminate"; shift ;;
    --dry-run) DRY=1; shift ;;
    --every) CHECK_EVERY="$2"; shift 2 ;;
    *) echo "Tham số lạ: $1" >&2; exit 1 ;;
  esac
done
[[ ${#PROJECTS[@]} -gt 0 ]] || { echo "Cần --project <tên bộ truyện>" >&2; exit 1; }

say() { echo "[$(date '+%H:%M:%S')] $*"; }

busy() {  # còn job nào của bộ này đang chạy?
  [[ "$(bash "${CODE_DIR}/run.sh" status --project "$1" 2>/dev/null)" == *PID* ]]
}

pending_videos() {  # video đã ghép nhưng chưa lên Drive (chưa có .done)
  local dir="${DATA_DIR}/projects/$1/output/video" n=0 f
  for f in "${dir}"/*.mp4; do
    [[ -e "$f" && ! -e "${f}.done" ]] && n=$((n + 1))
  done
  echo "$n"
}

say "Theo dõi: ${PROJECTS[*]} (kiểm tra mỗi ${CHECK_EVERY}s, xong thì ${ACTION} pod)"
for p in "${PROJECTS[@]}"; do
  while busy "$p"; do sleep "${CHECK_EVERY}"; done
  say "Bộ '$p': các job đã dừng. Ghép tập lẻ cuối + đẩy lại video còn sót..."
  # video --finish: ghép tập chưa đủ 20 chương (cuối truyện) và đẩy lại video upload lỗi.
  bash "${CODE_DIR}/run.sh" video --project "$p" --finish || say "Lệnh ghép cuối báo lỗi, kiểm tra lại."
done

# Kiểm tra lần cuối: mọi video đã lên Drive?
left=0
for p in "${PROJECTS[@]}"; do
  n="$(pending_videos "$p")"
  audio="$(ls "${DATA_DIR}/projects/$p/output/audio" 2>/dev/null | grep -c '\.wav$' || true)"
  done_eps="$(ls "${DATA_DIR}/projects/$p/output/video" 2>/dev/null | grep -c '\.done$' || true)"
  say "Bộ '$p': ${done_eps} tập đã lên Drive, ${n} video chưa lên, ${audio} audio chương còn trên pod."
  left=$((left + n))
done
if [[ "${left}" -gt 0 ]]; then
  say "CÒN ${left} video chưa lên Drive -> KHÔNG tắt pod (tránh mất video). Kiểm tra mạng/rclone rồi chạy lại script."
  exit 1
fi

POD_ID="${RUNPOD_POD_ID:-}"
if [[ -z "${POD_ID}" ]]; then
  say "Không thấy RUNPOD_POD_ID (không chạy trên RunPod?) -> không tắt."
  exit 0
fi
if [[ "${DRY}" == "1" ]]; then
  say "[dry-run] Đủ điều kiện ${ACTION} pod ${POD_ID}, nhưng không tắt."
  exit 0
fi

say "Mọi video đã lên Drive. ${ACTION} pod ${POD_ID} để ngừng tính tiền..."
if [[ "${ACTION}" == "terminate" ]]; then
  MUTATION="mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }"
  CTL=(remove pod "${POD_ID}")
else
  MUTATION="mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }"
  CTL=(stop pod "${POD_ID}")
fi
if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  # Khóa API đặt ở header (không đưa vào URL).
  if curl -fsS -X POST "https://api.runpod.io/graphql" \
       -H "Content-Type: application/json" -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
       -d "{\"query\": \"${MUTATION}\"}" | tee /dev/stderr | grep -vq '"errors"'; then
    say "Đã gửi lệnh ${ACTION} qua RunPod API."
    exit 0
  fi
  say "RunPod API chưa thực hiện được, thử runpodctl..."
fi
if command -v runpodctl >/dev/null; then
  runpodctl "${CTL[@]}" && { say "Đã gửi lệnh ${ACTION} qua runpodctl."; exit 0; }
fi
say "Không tự tắt được pod: hãy bấm Stop/Terminate trên RunPod."
exit 1
