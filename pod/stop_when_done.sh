#!/usr/bin/env bash
# Tự TẮT POD khi bộ truyện đã đọc xong, ghép xong và đẩy hết video lên Google Drive.
#
# Cách dùng thường ngày (tự bật kèm job, không cần chạy tay):
#   bash run.sh start --project "Đại Niết Bàn" --jobs 2 --stop-pod
#   (run.sh stop sẽ tắt luôn chế độ tự tắt pod -> dừng/chạy lại job thoải mái, không bị tắt nhầm)
#
# Chạy riêng:
#   nohup bash pod/stop_when_done.sh --project "Đại Niết Bàn" --jobs 2 > /workspace/stop_pod.log 2>&1 &
#   bash pod/stop_when_done.sh --project "A" --project "B"   # chờ nhiều bộ
#   bash pod/stop_when_done.sh --project "A" --terminate     # XÓA hẳn pod thay vì Stop
#   bash pod/stop_when_done.sh --project "A" --dry-run       # chỉ kiểm tra, không tắt
#
# Chỉ tắt khi THẬT SỰ xong:
#   1. Không còn job nào chạy, kiểm tra 2 lần liên tiếp (không tắt nhầm lúc đang dừng/chạy lại job).
#   2. Hỏi tool còn chương nào chưa đọc (studio.py progress). Còn -> job đã chết giữa chừng
#      (hết VRAM, lỗi mạng...) -> TỰ CHẠY LẠI job (tối đa --max-restarts lần) thay vì tắt pod.
#   3. Ghép tập lẻ cuối (--finish) + đẩy lại video upload lỗi.
#   4. Còn video .mp4 chưa lên Drive -> KHÔNG tắt (tránh mất video).
# Cách tắt giống AI_Video_Studio: RunPod API (RUNPOD_API_KEY có sẵn trong pod), dự phòng runpodctl.
set -Euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
[[ -f "${DATA_DIR}/env.sh" ]] && source "${DATA_DIR}/env.sh"
PYTHON="${AUDIO_STUDIO_PYTHON:-python3}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

PROJECTS=()
ACTION="stop"
DRY=0
CHECK_EVERY=120
JOBS=2
MAX_RESTARTS=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECTS+=("$2"); shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --terminate) ACTION="terminate"; shift ;;
    --dry-run) DRY=1; shift ;;
    --every) CHECK_EVERY="$2"; shift 2 ;;
    --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
    *) echo "Tham số lạ: $1" >&2; exit 1 ;;
  esac
done
[[ ${#PROJECTS[@]} -gt 0 ]] || { echo "Cần --project <tên bộ truyện>" >&2; exit 1; }

# Ghi PID để "run.sh stop" tắt được chế độ tự tắt pod.
echo "$$" > "${DATA_DIR}/stop_when_done.pid"
trap 'rm -f "${DATA_DIR}/stop_when_done.pid"' EXIT

say() { echo "[$(date '+%H:%M:%S')] $*"; }

busy() {  # còn job nào của bộ này đang chạy?
  # Chỉ đếm dòng job (audio1/2.., video); dòng "Tự tắt pod: BẬT (PID ..)" không tính.
  # Lấy cả trạng thái vào biến rồi mới tìm: "| grep -q" + pipefail -> SIGPIPE, vòng chờ thoát sai.
  local out
  out="$(bash "${CODE_DIR}/run.sh" status --project "$1" 2>/dev/null || true)"
  grep -Eq '^ +(audio[0-9]+|video): PID' <<<"${out}"
}

progress() {  # JSON tiến độ của tool: {"chapters":..,"chapters_pending":..,...}
  (cd "${CODE_DIR}" && "${PYTHON}" studio.py progress --project "$1" 2>/dev/null | tail -n 1)
}

json_get() {  # json_get '<json>' key
  "${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$1" "$2" 2>/dev/null
}

pending_videos() {  # video đã ghép nhưng chưa lên Drive (chưa có .done)
  local dir="${DATA_DIR}/projects/$1/output/video" n=0 f
  for f in "${dir}"/*.mp4; do
    [[ -e "$f" && ! -e "${f}.done" ]] && n=$((n + 1))
  done
  echo "$n"
}

say "Theo dõi: ${PROJECTS[*]} (kiểm tra mỗi ${CHECK_EVERY}s; xong thì ${ACTION} pod; job chết giữa chừng thì chạy lại tối đa ${MAX_RESTARTS} lần)"
for p in "${PROJECTS[@]}"; do
  idle=0
  restarts=0
  while true; do
    if busy "$p"; then
      idle=0
      sleep "${CHECK_EVERY}"
      continue
    fi
    idle=$((idle + 1))
    if [[ "${idle}" -lt 2 ]]; then      # nghỉ thêm 1 vòng: có thể đang dừng/chạy lại job bằng tay
      sleep "${CHECK_EVERY}"
      continue
    fi
    info="$(progress "$p")"
    left="$(json_get "${info}" chapters_pending)"
    if [[ -z "${left}" ]]; then
      say "Bộ '$p': không đọc được tiến độ (${info:-trống}); coi như xong."
      break
    fi
    if [[ "${left}" -gt 0 ]]; then
      if [[ "${restarts}" -lt "${MAX_RESTARTS}" ]]; then
        restarts=$((restarts + 1))
        say "Bộ '$p': các job đã dừng nhưng còn ${left} chương chưa đọc (từ chương $(json_get "${info}" first_pending)) -> chạy lại lần ${restarts}/${MAX_RESTARTS}."
        bash "${CODE_DIR}/run.sh" start --project "$p" --jobs "${JOBS}" || say "Chạy lại báo lỗi."
        idle=0
        sleep "${CHECK_EVERY}"
        continue
      fi
      say "Bộ '$p': đã chạy lại ${MAX_RESTARTS} lần vẫn còn ${left} chương -> bỏ cuộc, vẫn tắt pod cho đỡ tốn tiền (xem log lỗi trong work/logs). Pod sau sẽ làm tiếp từ chỗ dở."
    else
      say "Bộ '$p': đã đọc hết $(json_get "${info}" chapters) chương ($(json_get "${info}" episodes_done)/$(json_get "${info}" episodes) tập đã lên Drive)."
    fi
    break
  done
  say "Bộ '$p': ghép tập lẻ cuối + đẩy lại video còn sót..."
  # video --finish: ghép tập chưa đủ chương (cuối truyện) và đẩy lại video upload lỗi.
  bash "${CODE_DIR}/run.sh" video --project "$p" --finish || say "Lệnh ghép cuối báo lỗi, kiểm tra lại."
done

# Kiểm tra lần cuối: mọi video đã lên Drive?
left=0
for p in "${PROJECTS[@]}"; do
  n="$(pending_videos "$p")"
  done_eps="$(ls "${DATA_DIR}/projects/$p/output/video" 2>/dev/null | grep -c '\.done$' || true)"
  say "Bộ '$p': ${done_eps} tập đã lên Drive, ${n} video chưa lên."
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
