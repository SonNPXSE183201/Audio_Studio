#!/usr/bin/env bash
# Chạy Audio Studio bằng đúng môi trường của pod.
#   bash run.sh check                 kiểm tra môi trường
#   bash run.sh start [--range 1-400] chạy nền (đọc audio + ghép tập), tắt terminal vẫn chạy
#   bash run.sh bench --price 0.69    đo tốc độ + chi phí thật của pod (~5 phút)
#   bash run.sh log                   xem log đang chạy
#   bash run.sh stop                  dừng tiến trình chạy nền
#   bash run.sh run|audio|video ...   chạy trực tiếp (giữ terminal)
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
ENV_FILE="${DATA_DIR}/env.sh"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${AUDIO_STUDIO_PYTHON:-python3}"
if [[ ! -x "${PYTHON}" ]] && ! command -v "${PYTHON}" >/dev/null; then
  echo "Chưa cài môi trường. Chạy: bash ${PROJECT_DIR}/pod/setup_pod.sh" >&2
  exit 1
fi

PID_FILE="${DATA_DIR}/work/studio.pid"
LOG_FILE="${DATA_DIR}/work/studio_run.log"
cmd="${1:-}"
shift || true

case "${cmd}" in
  start)
    mkdir -p "$(dirname "${LOG_FILE}")"
    if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "[OK] Đang chạy (PID $(cat "${PID_FILE}")). Xem: bash run.sh log"
      exit 0
    fi
    cd "${PROJECT_DIR}"
    nohup "${PYTHON}" -u studio.py run "$@" >>"${LOG_FILE}" 2>&1 </dev/null &
    echo "$!" >"${PID_FILE}"
    sleep 2
    if kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "[OK] Đang chạy nền (PID $(cat "${PID_FILE}")). Xem: bash run.sh log"
    else
      echo "[LỖI] Không khởi động được:" >&2
      tail -n 40 "${LOG_FILE}" >&2
      exit 1
    fi
    ;;
  log)
    exec tail -n 100 -f "${LOG_FILE}"
    ;;
  stop)
    if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill -TERM "$(cat "${PID_FILE}")" && echo "[OK] Đã dừng."
    else
      echo "Không có tiến trình nào đang chạy."
    fi
    ;;
  run|audio|video|check|bench)
    cd "${PROJECT_DIR}"
    exec "${PYTHON}" -u studio.py "${cmd}" "$@"
    ;;
  *)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
