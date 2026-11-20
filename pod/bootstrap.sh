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
