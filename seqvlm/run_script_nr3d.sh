#!/bin/bash

cd /root/SeqVLM-clean/seqvlm

export PYTHONPATH=/root/SeqVLM-clean:$PYTHONPATH

# 防止线程占用过高
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u evaluate_nr3d.py \
  --data_path ../data/nr3d_250.json \
  --exp_name dynamic_canvas_nr3d_full \
  --image_path ../data/nr3d_preprocessed \
  --vlm_model qwen \
  --max_vlm_props 40 \
  --max_batch_size 4 \
  --seed 42 \
  --use_anchor_aware \
  --use_dynamic_canvas \
  --crop_image_root ../data/crop_images_nr3d \
  --crop_pool_meta_root ../data/crop_pool_meta_nr3d \
  --view_meta_root ../data/view_meta_nr3d \
  --posed_image_root ../data/posed_images_rgb_pose \
  --dynamic_canvas_root ../data/dynamic_canvas_nr3d_anchor085 \
  --canvas_k 5 \
  --num_appearance_views 2 \
  --num_relation_views 2 \
  --use_global_context\
  --use_final_global_view \
  --global_rendered_root ../data/global_rendered_views_nr3d \
  --final_global_view_root ../data/final_global_aux_nr3d \
  --max_final_global_anchors 2\
  --seg_conf_score 0.85 \