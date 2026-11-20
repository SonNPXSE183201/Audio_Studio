#!/usr/bin/env bash
# MỘT LỆNH cho pod mới tạo hoặc pod vừa Start lại:
#
#   curl -fsSL https://raw.githubusercontent.com/SonNPXSE183201/Audio_Studio/main/pod/bootstrap.sh | bash
#
# - Code đặt ở /root/Audio_Studio (ổ container): mất khi Stop pod -> lệnh này tự clone lại.
# - Dữ liệu (chương, giọng mẫu, audio, video, model) nằm ở /workspace/audio_studio: không mất.
set -Eeuo pipefail

REPO="${REPO:-https://github.com/SonNPXSE183201/Audio_Studio.git}"
CODE_DIR="${CODE_DIR:-/root/Audio_Studio}"
LOG="${SETUP_LOG:-/workspace/audio_studio_setup.log}"

if [[ -d "${CODE_DIR}/.git" ]]; then
  git -C "${CODE_DIR}" pull --ff-only
else
  git clone --depth 1 "${REPO}" "${CODE_DIR}"
fi
mkdir -p "$(dirname "${LOG}")"
echo "[INFO] Cài đặt, log đầy đủ: ${LOG}"
bash "${CODE_DIR}/pod/setup_pod.sh" 2>&1 | tee "${LOG}"

# AUTO_RUN=1: cài xong tự chạy nền (lấy chương từ Drive -> đọc -> ghép tập -> upload).
#   PROJECTS="Đại Niết Bàn;Truyen B"  các bộ truyện (đúng tên thư mục trên Drive), NGĂN CÁCH BẰNG ;
#                                     chạy lần lượt từng bộ
#   JOBS=2                         số job đọc audio mỗi bộ (dùng chung GPU)
if [[ "${AUTO_RUN:-0}" == "1" ]]; then
  if [[ -n "${PROJECTS:-}" ]]; then
    # Các bộ chạy NỐI TIẾP nhau trong nền (chung GPU, không tranh nhau).
    # Tên bộ có thể có dấu cách (vd "Đại Niết Bàn") -> các bộ ngăn cách bằng dấu ;
    IFS=';' read -r -a LIST <<<"${PROJECTS}"
    CLEAN=()
    for p in "${LIST[@]}"; do
      p="$(echo "$p" | sed 's/^ *//;s/ *$//')"
      [[ -n "$p" ]] && CLEAN+=("$p")
    done
    # shellcheck disable=SC2086
    nohup bash "${CODE_DIR}/pod/run_projects.sh" --jobs "${JOBS:-1}" ${RUN_ARGS:-} "${CLEAN[@]}" \
      >/workspace/audio_studio_projects.log 2>&1 </dev/null &
    echo "[OK] Chạy lần lượt các bộ: ${PROJECTS}. Xem: bash ${CODE_DIR}/run.sh log --project <tên>"
  else
    bash "${CODE_DIR}/run.sh" start --jobs "${JOBS:-1}" ${RUN_ARGS:-}
  fi
fi
