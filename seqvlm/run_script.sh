export PYTHONPATH=$(dirname "$PWD")
mkdir -p ../logs
python evaluate.py \
--exp_name visprog_scanrefer \
--image_path ../data/scanrefer_preprocessed \
--vlm_model qwen