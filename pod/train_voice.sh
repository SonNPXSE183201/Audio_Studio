#!/usr/bin/env bash
# Fine-tune LoRA giọng riêng cho VieNeu-TTS v3 Turbo trên pod, rồi chấm điểm so với clone tức thì.
#
#   bash pod/train_voice.sh <dataset> "<Tên giọng>" <giọng mẫu.wav> <chương thử.txt> [epoch ...]
#
#   <dataset>  thư mục có metadata.csv + raw_audio/ (finetune/build_voice_dataset.py tạo ra),
#              hoặc đường dẫn Drive "gdrive:..." (tự tải về).
#   epoch      mặc định "2 3 5": train mỗi mức một model để chọn bản giống giọng nhất mà chưa overfit.
#   Biến môi trường: LORA_ARGS="--target all --r 32" (tham số thêm cho train_lora.py), RUN_TAG="_all"
#   (hậu tố tên run), EVAL_OUT (thư mục kết quả chấm điểm).
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
git -C "${VIENEU_DIR}" checkout -q -f "${VIENEU_COMMIT}"
# Kiểm tra issue #198 của prepare_dataset.py chặn khi > 50% clip kết thúc cùng mã codebook-0, để bắt
# khung đệm lỗi 455 (encoder cũ). Với audio thật cắt tự nhiên, khung cuối gần như luôn là im lặng
# (encoder đệm 0 cho đủ 80 ms) -> mã 482 = mã IM LẶNG (đo: 0,5 s im lặng -> toàn 482) -> chặn nhầm.
# Chỉ bỏ qua 482; 455 vẫn bị chặn.
sed -i 's/    if len(recs) >= 20 and n_same \/ len(recs) > 0.5:/    if len(recs) >= 20 and n_same \/ len(recs) > 0.5 and code != 482:  # 482 = im lặng/' \
    "${VIENEU_DIR}/finetune/prepare_dataset.py"
grep -q "code != 482" "${VIENEU_DIR}/finetune/prepare_dataset.py" || { echo "Không vá được prepare_dataset.py"; exit 1; }
"${PY}" -m pip install -q "peft>=0.13" pyarrow accelerate safetensors "torchmetrics>=1.4" librosa requests

if [[ "${DATASET}" == gdrive:* ]]; then
  log "Tải dataset từ Drive: ${DATASET}"
  rclone copy --config "${DATA_DIR}/rclone.conf" "${DATASET}" "${WORK}/dataset"
  DATASET="${WORK}/dataset"
fi
[[ -f "${DATASET}/metadata.csv" ]] || { echo "Không thấy ${DATASET}/metadata.csv"; exit 1; }

# Script fine-tune của VieNeu chỉ đọc/ghi BÊN TRONG thư mục đang đứng (safe_path): chạy từ ${WORK}
# và đưa mọi file vào trong đó.
inside() { [[ "$(realpath "$1")" == "$(realpath "${WORK}")"/* ]]; }
if ! inside "${DATASET}"; then
  log "Chép dataset vào ${WORK}/dataset"
  mkdir -p "${WORK}/dataset" && cp -r "${DATASET}"/. "${WORK}/dataset/"
  DATASET="${WORK}/dataset"
fi
# Clip đóng gói giọng / clone đối chứng (3-8 s, sạch); mặc định là chính giọng mẫu.
CLIP="${VOICE_CLIP:-${REF}}"
if ! inside "${CLIP}"; then
  cp "${CLIP}" "${WORK}/clip_dong_goi_giong.wav"
  CLIP="${WORK}/clip_dong_goi_giong.wav"
fi
DATASET="$(realpath "${DATASET}")"; CLIP="$(realpath "${CLIP}")"

cd "${WORK}"
if [[ -f "${DATASET}/train.parquet" && "${DATASET}/train.parquet" -nt "${DATASET}/metadata.csv" ]]; then
  log "2/5 - Dùng lại ${DATASET}/train.parquet (mới hơn metadata.csv)"
else
  log "2/5 - Mã hóa dataset ($(wc -l < "${DATASET}/metadata.csv") clip)"
  "${PY}" "${VIENEU_DIR}/finetune/prepare_dataset.py" --dataset-dir "${DATASET}"
fi

MODELS=(--model "goc=clone:${CLIP}")
for E in "${EPOCHS[@]}"; do
  RUN="${SLUG}${RUN_TAG:-}_e${E}"
  log "3/5 - Train LoRA ${E} epoch ${LORA_ARGS:-} -> ${RUN}"
  # Dataset nhỏ (~200-300 clip): eval mỗi 10 bước để thấy lúc eval loss ngừng giảm (overfit).
  "${PY}" "${VIENEU_DIR}/finetune/train_lora.py" --data "${DATASET}/train.parquet" --run "${RUN}" \
      --output-dir "${WORK}/output" --epochs "${E}" --eval-ratio 0.05 --eval-every 10 \
      --save-every 100000 --merge ${LORA_ARGS:-} 2>&1 | tee "${WORK}/${RUN}.log"
  log "4/5 - Đóng gói giọng '${VOICE}' vào ${RUN}/merged"
  "${PY}" "${VIENEU_DIR}/finetune/make_voice.py" --audio "${CLIP}" --name "${VOICE}" --default \
      --out "${WORK}/output/${RUN}/merged"
  # Thêm giọng từ hồ sơ TRUNG BÌNH cả dataset (thường đồng đều hơn một clip lẻ) để chấm so sánh.
  "${PY}" "${PROJECT_DIR}/finetune/make_mean_voice.py" --data "${DATASET}/train.parquet"       --name "${VOICE} TB" --out "${WORK}/output/${RUN}/merged"
  MODELS+=(--model "${RUN_TAG:-}e${E}=${WORK}/output/${RUN}/merged:${VOICE}")
  MODELS+=(--model "${RUN_TAG:-}e${E}tb=${WORK}/output/${RUN}/merged:${VOICE} TB")
done

log "5/5 - Chấm điểm: clone tức thì vs các model fine-tune"
cd "${PROJECT_DIR}"
"${PY}" finetune/eval_voice.py --ref "${REF}" --chapter "${CHAPTER}" --out "${EVAL_OUT:-${WORK}/eval}" "${MODELS[@]}"

cat <<EOF

[XONG] Chọn model có "giống giọng" cao nhất mà CER không tăng, rồi đặt trong config.yaml:
  tts:
    backbone_repo: "${WORK}/output/${SLUG}_e<N>/merged"
    voice: "${VOICE}"
Nghe thử: ${WORK}/eval/*.wav
EOF
