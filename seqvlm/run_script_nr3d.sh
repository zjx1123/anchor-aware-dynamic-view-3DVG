export PYTHONPATH=$(dirname "$PWD")
mkdir -p ../logs
python evaluate_nr3d.py \
--exp_name visprog_nr3d \
--image_path ../data/nr3d_preprocessed \
--vlm_model qwen