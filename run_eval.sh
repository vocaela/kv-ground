# KV-Ground-4B-BaseQw3vl, KV-Ground-4B-BaseGuiOwl1.5, or Qwen3-VL-4B local path
MODEL_DIR="/path/to/model"
OUTPUT_DIR="/path/to/output"

# ScreenSpot-Pro
# The Screenspot-Pro data downloaded from https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/main
SSPRO_DATA_DIR="/path/to/screenspot-pro-data"
torchrun --nproc_per_node=8 eval_sspro_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$SSPRO_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \

# ScreenSpot-v2
# The Screenspot-v2 data downloaded from https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2
# Note that need first unzip screenspotv2_image.zip into a folder "screenspotv2_images"
SSV2_DATA_DIR="/path/to/screenspot-v2-data"
torchrun --nproc_per_node=8 eval_ssv2_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$SSV2_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \

# OSWorld-G
# The OSWorldG data downloaded from https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark
OSWORLDG_DATA_DIR="/path/to/osworldg-data"
torchrun --nproc_per_node=8 eval_osworldg_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$OSWORLDG_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \

# OSWorld-G-refined, the same OSWorldG data downloaded from https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark, flag '--refined true' indicates it uses the 'OSWorld-G_refined.json' dataset file
torchrun --nproc_per_node=8 eval_osworldg_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$OSWORLDG_DATA_DIR" \
  --refined true \
  --output_dir "$OUTPUT_DIR" \

# UI-vision
# The ui-vision data downloaded from https://huggingface.co/datasets/ServiceNow/ui-vision/tree/main
UIVISION_DATA_DIR="/path/to/uivision-data"
torchrun --nproc_per_node=8 eval_uivision_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$UIVISION_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \