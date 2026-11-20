#!/usr/bin/env bash
# Chạy Audio Studio bằng đúng môi trường của pod.
#   bash run.sh check  [--project X]                 kiểm tra môi trường
#   bash run.sh start  [--project X] [--jobs 2] ...  chạy nền: N job đọc audio + 1 job ghép video
#   bash run.sh status [--project X]                 các job đang chạy, tiến độ
#   bash run.sh log    [--project X]                 xem log (Ctrl+C thoát xem, job vẫn chạy)
#   bash run.sh stop   [--project X]                 dừng các job của bộ truyện
#   bash run.sh bench --price 0.69                   đo tốc độ + chi phí thật của pod (~5 phút)
#   bash run.sh run|audio|video ...                  chạy trực tiếp (giữ terminal)
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
ENV_FILE="${DATA_DIR}/env.sh"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${AUDIO_STUDIO_PYTHON:-python3}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
if [[ ! -x "${PYTHON}" ]] && ! command -v "${PYTHON}" >/dev/null; then
  echo "Chưa cài môi trường. Chạy: bash ${PROJECT_DIR}/pod/setup_pod.sh" >&2
  exit 1
fi

cmd="${1:-}"
shift || true

# Tách --project và --jobs ra khỏi các tham số còn lại.
PROJECT=""
JOBS=1
REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --project=*) PROJECT="${1#*=}"; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    *) REST+=("$1"); shift ;;
  esac
done
PROJECT_ARGS=()
WORK_DIR="${DATA_DIR}/work"
if [[ -n "${PROJECT}" ]]; then
  PROJECT_ARGS=(--project "${PROJECT}")
  WORK_DIR="${DATA_DIR}/projects/${PROJECT}/work"
fi
PID_DIR="${WORK_DIR}/pids"
LOG_DIR="${WORK_DIR}/logs"

running_pids() {
  local f
  for f in "${PID_DIR}"/*.pid; do
    [[ -e "$f" ]] || continue
    if kill -0 "$(cat "$f")" 2>/dev/null; then echo "$f"; else rm -f "$f"; fi
  done
}

launch() {  # launch <tên> <tham số studio.py...>
  local name="$1"; shift
  nohup "${PYTHON}" -u studio.py "$@" >>"${LOG_DIR}/run_${name}.log" 2>&1 </dev/null &
  echo "$!" >"${PID_DIR}/${name}.pid"
  echo "[OK] ${name}: PID $!"
}

case "${cmd}" in
  start)
    mkdir -p "${PID_DIR}" "${LOG_DIR}"
    if [[ -n "$(running_pids)" ]]; then
      echo "[OK] Bộ truyện '${PROJECT:-mặc định}' đang chạy. Xem: bash run.sh status ${PROJECT:+--project \"${PROJECT}\"}"
      exit 0
    fi
    cd "${PROJECT_DIR}"
    # Nhiều job dùng chung 1 GPU -> mỗi job tự giảm batch.
    export AUDIO_STUDIO_GPU_JOBS="${JOBS}"
    for i in $(seq 1 "${JOBS}"); do
      job="audio$(printf '%d' "$i")"
      launch "${job}" audio "${PROJECT_ARGS[@]}" --job "${job}" "${REST[@]}"
      sleep 1
    done
    # Job ghép video chạy bằng CPU: đủ chương là ghép tập + upload, tự dừng khi audio xong hết.
    VIDEO_ARGS=()
    for a in "${REST[@]}"; do [[ "$a" == "--finish" ]] && VIDEO_ARGS+=(--finish); done
    launch video video --watch "${PROJECT_ARGS[@]}" --job video "${VIDEO_ARGS[@]}"
    echo "Xem tiến độ: bash run.sh log ${PROJECT:+--project \"${PROJECT}\"}"
    ;;
  status)
    pids="$(running_pids)"
    if [[ -z "${pids}" ]]; then echo "Không có job nào đang chạy."; else
      while IFS= read -r f; do echo "  $(basename "$f" .pid): PID $(cat "$f")"; done <<<"${pids}"
    fi
    base="${DATA_DIR}${PROJECT:+/projects/${PROJECT}}"
    echo "  Chương: $(find "${base}/input" -name 'Chuong_*.txt' 2>/dev/null | wc -l) | audio xong: $(ls "${base}/output/audio" 2>/dev/null | grep -c '\.wav$' || true) | tập xong: $(ls "${base}/output/video" 2>/dev/null | grep -c '\.done$' || true)"
    ;;
  log)
    mkdir -p "${LOG_DIR}"
    exec tail -n 50 -f "${LOG_DIR}"/run_*.log
    ;;
  stop)
    pids="$(running_pids)"
    if [[ -z "${pids}" ]]; then echo "Không có job nào đang chạy."; exit 0; fi
    while IFS= read -r f; do kill -TERM "$(cat "$f")" && rm -f "$f" && echo "[OK] Đã dừng $(basename "$f" .pid)"; done <<<"${pids}"
    ;;
  run|audio|video|check|bench)
    cd "${PROJECT_DIR}"
    exec "${PYTHON}" -u studio.py "${cmd}" "${PROJECT_ARGS[@]}" "${REST[@]}"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
