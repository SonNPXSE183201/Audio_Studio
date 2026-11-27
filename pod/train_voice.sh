#!/usr/bin/env bash
# Fine-tune LoRA giọng riêng cho VieNeu-TTS v3 Turbo trên pod, rồi chấm điểm so với clone tức thì.
#
#   bash pod/train_voice.sh <dataset> "<Tên giọng>" <giọng mẫu.wav> <chương thử.txt> [epoch ...]
#
#   <dataset>  thư mục có metadata.csv + raw_audio/ (finetune/build_voice_dataset.py tạo ra),
#              hoặc đường dẫn Drive "gdrive:..." (tự tải về).
#   epoch      mặc định "2 3 5": train mỗi mức một model để chọn bản giống giọng nhất mà chưa overfit.
#
# Kết quả: ${AUDIO_STUDIO_DATA}/voice_lora/output/<giọng>_e<N>/merged  (model + giọng đóng gói)
#          ${AUDIO_STUDIO_DATA}/voice_lora/eval/ket_qua.json + file nghe thử từng model.
set -Eeuo pipefail

DATASET="${1:?thiếu thư mục dataset}"
VOICE="${2:?thiếu tên giọng}"
REF="${3:?thiếu giọng mẫu .wav}"
CHAPTER="${4:?thiếu chương thử .txt}"
shift 4
EPOCHS=("$@")
[[ ${#EPOCHS[@]} -gt 0 ]] || EPOCHS=(2 3 5)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AUDIO_STUDIO_DATA:-/workspace/audio_studio}"
[[ -f "${DATA_DIR}/env.sh" ]] && source "${DATA_DIR}/env.sh"
PY="${AUDIO_STUDIO_PYTHON:-python3}"
WORK="${DATA_DIR}/voice_lora"
VIENEU_DIR="${WORK}/VieNeu-TTS"
# Cùng commit với gói vieneu==3.8.5 mà tool ghim: dataset/model khớp đúng cách tool suy luận.
VIENEU_COMMIT="2e3532fd65c00fc1862cf8ba89f309899453b4f7"
SLUG="$(echo "${VOICE}" | iconv -f utf-8 -t ascii//TRANSLIT 2>/dev/null | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//' | tr 'A-Z' 'a-z')"
SLUG="${SLUG:-giong}"
log() { echo -e "\n[TRAIN] $*"; }
mkdir -p "${WORK}"

log "1/5 - Mã nguồn fine-tune của VieNeu (commit ${VIENEU_COMMIT:0:7})"
if [[ ! -d "${VIENEU_DIR}/.git" ]]; then
  git clone -q https://github.com/pnnbao97/VieNeu-TTS.git "${VIENEU_DIR}"
fi
git -C "${VIENEU_DIR}" fetch -q origin "${VIENEU_COMMIT}" 2>/dev/null || true
git -C "${VIENEU_DIR}" checkout -q "${VIENEU_COMMIT}"
"${PY}" -m pip install -q "peft>=0.13" pyarrow accelerate safetensors

if [[ "${DATASET}" == gdrive:* ]]; then
  log "Tải dataset từ Drive: ${DATASET}"
  rclone copy --config "${DATA_DIR}/rclone.conf" "${DATASET}" "${WORK}/dataset"
  DATASET="${WORK}/dataset"
fi
[[ -f "${DATASET}/metadata.csv" ]] || { echo "Không thấy ${DATASET}/metadata.csv"; exit 1; }

cd "${VIENEU_DIR}"
log "2/5 - Mã hóa dataset ($(wc -l < "${DATASET}/metadata.csv") clip)"
"${PY}" finetune/prepare_dataset.py --dataset-dir "${DATASET}"

MODELS=(--model "goc=clone:${REF}")
for E in "${EPOCHS[@]}"; do
  RUN="${SLUG}_e${E}"
  log "3/5 - Train LoRA ${E} epoch -> ${RUN}"
  # Dataset nhỏ (~200-300 clip): eval mỗi 10 bước để thấy lúc eval loss ngừng giảm (overfit).
  "${PY}" finetune/train_lora.py --data "${DATASET}/train.parquet" --run "${RUN}" \
      --output-dir "${WORK}/output" --epochs "${E}" --eval-ratio 0.05 --eval-every 10 \
      --save-every 100000 --merge 2>&1 | tee "${WORK}/${RUN}.log"
  log "4/5 - Đóng gói giọng '${VOICE}' vào ${RUN}/merged"
  "${PY}" finetune/make_voice.py --audio "${REF}" --name "${VOICE}" --default \
      --out "${WORK}/output/${RUN}/merged"
  MODELS+=(--model "e${E}=${WORK}/output/${RUN}/merged:${VOICE}")
done

log "5/5 - Chấm điểm: clone tức thì vs các model fine-tune"
cd "${PROJECT_DIR}"
"${PY}" finetune/eval_voice.py --ref "${REF}" --chapter "${CHAPTER}" --out "${WORK}/eval" "${MODELS[@]}"

cat <<EOF

[XONG] Chọn model có "giống giọng" cao nhất mà CER không tăng, rồi đặt trong config.yaml:
  tts:
    backbone_repo: "${WORK}/output/${SLUG}_e<N>/merged"
    voice: "${VOICE}"
Nghe thử: ${WORK}/eval/*.wav
EOF
