#!/bin/bash

set -e

cd "$(dirname "$0")"

PROJECT_ROOT="$(cd .. && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python evaluate.py \
  --data_path ../data/scanrefer_250.json \
  --exp_name dynamic_canvas_scanrefer_250 \
  --image_path ../data/scanrefer_preprocessed \
  --vlm_model qwen \
  --max_vlm_props 40 \
  --use_dynamic_canvas \
  --crop_image_root ../data/crop_images \
  --crop_pool_meta_root ../data/crop_pool_meta_scanrefer \
  --view_meta_root ../data/view_meta_scanrefer \
  --posed_image_root ../data/posed_images_rgb_pose \
  --dynamic_canvas_root ../data/dynamic_canvas_scanrefer \
  --canvas_k 5\
  --use_final_global_view \
  --global_rendered_root ../data/global_rendered_views_scanrefer_spazer \
  --final_global_view_root ../data/final_global_aux_scanrefer \
  --max_final_global_anchors 2