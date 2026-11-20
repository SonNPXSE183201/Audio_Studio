#!/usr/bin/env bash
# Chạy NỐI TIẾP nhiều bộ truyện trên một pod (chung GPU, không tranh nhau):
#   bash pod/run_projects.sh --jobs 2 Truyen_A Truyen_B Truyen_C
# Mỗi bộ: N job đọc audio + 1 job ghép video; bộ này xong hết mới sang bộ sau.
set -Euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS=1
EXTRA=()
PROJECTS=()
STOP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --stop-pod) STOP="stop"; shift ;;           # xong hết + mọi video đã lên Drive -> Stop pod
    --terminate-pod) STOP="terminate"; shift ;; # ... -> xóa hẳn pod
    --*) EXTRA+=("$1"); shift ;;
    *) PROJECTS+=("$1"); shift ;;
  esac
done

for project in "${PROJECTS[@]}"; do
  echo "[$(date +%H:%M:%S)] === Bắt đầu bộ: ${project} (${JOBS} job) ==="
  bash "${CODE_DIR}/run.sh" start --project "${project}" --jobs "${JOBS}" "${EXTRA[@]}"
  # Không dùng "| grep -q": grep thoát sớm -> SIGPIPE + pipefail làm vòng chờ dừng ngay.
  while [[ "$(bash "${CODE_DIR}/run.sh" status --project "${project}" 2>/dev/null)" == *PID* ]]; do
    sleep 60
  done
  echo "[$(date +%H:%M:%S)] === Xong bộ: ${project} ==="
done
echo "[$(date +%H:%M:%S)] Đã chạy xong mọi bộ truyện."

if [[ -n "${STOP}" ]]; then
  ARGS=()
  for project in "${PROJECTS[@]}"; do ARGS+=(--project "${project}"); done
  [[ "${STOP}" == "terminate" ]] && ARGS+=(--terminate)
  # Ghép tập lẻ cuối, đẩy nốt video, chỉ tắt khi mọi video đã lên Drive.
  bash "${CODE_DIR}/pod/stop_when_done.sh" --every 30 "${ARGS[@]}"
fi
