#!/usr/bin/env bash
# Cài môi trường Audio Studio trên RunPod (template PyTorch có sẵn CUDA).
# Chạy lại sau mỗi lần Start pod: phần trên volume /workspace (model, dữ liệu) được giữ,
# chỉ cài lại gói Python vào ổ container (mất ~2-4 phút).
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
VENV="${AUDIO_STUDIO_VENV:-/root/audio-studio-venv}"
ENV_FILE="${DATA_DIR}/env.sh"

log() { echo -e "\n[SETUP] $*"; }

log "1/5 - Gói hệ thống (ffmpeg, rclone, libsndfile)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends ffmpeg libsndfile1 rclone curl git >/dev/null
ffmpeg -version | head -1

log "2/5 - Python venv (dùng lại PyTorch CUDA có sẵn của template)"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/python" -m pip install -q --upgrade pip wheel
"${VENV}/bin/python" -m pip install -q -r "${PROJECT_DIR}/requirements.txt"
"${VENV}/bin/python" -m pip install -q omnivoice transformers accelerate sentencepiece huggingface_hub

log "3/5 - Kiểm tra GPU"
"${VENV}/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch không thấy GPU: chọn template RunPod PyTorch (CUDA)"
props = torch.cuda.get_device_properties(0)
print(f"[OK] PyTorch {torch.__version__}; GPU {props.name} {props.total_memory / 1024**3:.0f} GB")
PY

log "4/5 - Thư mục dữ liệu trên volume: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"/{input,output/audio,output/video,work,assets/voice,assets/background,Kenh,.cache/huggingface}
[[ -f "${DATA_DIR}/config.yaml" ]] || cp "${PROJECT_DIR}/config.yaml" "${DATA_DIR}/config.yaml"
# Giọng mẫu có sẵn trong code (nếu có) được chép sang volume lần đầu.
for f in "${PROJECT_DIR}"/assets/voice/*; do
  [[ -e "$f" ]] || continue
  [[ -f "${DATA_DIR}/assets/voice/$(basename "$f")" ]] || cp "$f" "${DATA_DIR}/assets/voice/"
done
cat > "${ENV_FILE}" <<EOF
export AUDIO_STUDIO_DATA="${DATA_DIR}"
export AUDIO_STUDIO_PYTHON="${VENV}/bin/python"
export HF_HOME="${DATA_DIR}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
# shellcheck disable=SC1090
source "${ENV_FILE}"

log "5/5 - Tải trước model (lưu trên volume, lần sau không tải lại)"
"${VENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("k2-fsa/OmniVoice", "vinai/PhoWhisper-large"):
    try:
        snapshot_download(repo)
        print(f"[OK] {repo}")
    except Exception as exc:
        print(f"[CẢNH BÁO] Không tải được {repo}: {exc}")
PY

cat <<EOF

[XONG] Cài đặt hoàn tất.
  Dữ liệu:  ${DATA_DIR}
    input/                 chép các file Chuong_XXXX.txt vào đây
    assets/voice/          giọng mẫu .wav + .txt (lời đọc khớp với giọng mẫu)
    assets/background/     background.mp4 (video nền 1-2 phút)
    Kenh/                  Script_Intro.mp3, Script_Watermark.mp3, Script_Watermark1.mp3
    config.yaml            cấu hình (sửa ở đây, cập nhật code không ghi đè)
    rclone.conf            cấu hình Google Drive (nếu upload)
  Kiểm tra:  bash ${PROJECT_DIR}/run.sh check
  Chạy nền:  bash ${PROJECT_DIR}/run.sh start
EOF
