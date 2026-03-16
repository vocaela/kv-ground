# Local path for
# KV-Ground-4B-BaseQw3vl, KV-Ground-4B-BaseGuiOwl1.5-0228, KV-Ground-8B-BaseGuiOwl1.5-0315, Gui-Owl-1.5-4B-Instruct, Gui-Owl-1.5-8B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct
MODEL_DIR="/path/to/model"

# output paths
SSPRO_OUTPUT_DIR="/path/to/sspro/output"
SSPRO_ZOOMIN_OUTPUT_DIR="/path/to/sspro/zoomin/output"
SSV2_OUTPUT_DIR="/path/to/ssv2/output"
OSWORLDG_OUTPUT_DIR="/path/to/osworldg/output"
OSWORLDG_REFINED_OUTPUT_DIR="/path/to/osworldg_refined/output"
UIVISION_OUTPUT_DIR="/path/to/uivision/output"

# ScreenSpot-Pro
# The Screenspot-Pro data downloaded from https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/main
SSPRO_DATA_DIR="/path/to/screenspot-pro-data"
torchrun --nproc_per_node=8 eval_sspro_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$SSPRO_DATA_DIR" \
  --output_dir "$SSPRO_OUTPUT_DIR" \

# Run Zoom-In for the 2nd round evaluation
torchrun --nproc_per_node=8 eval_sspro_zoomin_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$SSPRO_DATA_DIR" \
  --output_dir "$SSPRO_ZOOMIN_OUTPUT_DIR" \
  --step1_output_dir "$SSPRO_OUTPUT_DIR" \

# ScreenSpot-v2
# The Screenspot-v2 data downloaded from https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2
# Note that need first unzip screenspotv2_image.zip into a folder "screenspotv2_images"
SSV2_DATA_DIR="/path/to/screenspot-v2-data"
torchrun --nproc_per_node=8 eval_ssv2_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$SSV2_DATA_DIR" \
  --output_dir "$SSV2_OUTPUT_DIR" \

# OSWorld-G
# The OSWorldG data downloaded from https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark
OSWORLDG_DATA_DIR="/path/to/osworldg-data"
torchrun --nproc_per_node=8 eval_osworldg_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$OSWORLDG_DATA_DIR" \
  --output_dir "$OSWORLDG_OUTPUT_DIR" \

# OSWorld-G-refined, the same OSWorldG data downloaded from https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark, flag '--refined true' indicates it uses the 'OSWorld-G_refined.json' dataset file
torchrun --nproc_per_node=8 eval_osworldg_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$OSWORLDG_DATA_DIR" \
  --refined true \
  --output_dir "$OSWORLDG_REFINED_OUTPUT_DIR" \

# UI-vision
# The ui-vision data downloaded from https://huggingface.co/datasets/ServiceNow/ui-vision/tree/main
UIVISION_DATA_DIR="/path/to/uivision-data"
torchrun --nproc_per_node=8 eval_uivision_hf_dp.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$UIVISION_DATA_DIR" \
  --output_dir "$UIVISION_OUTPUT_DIR" \