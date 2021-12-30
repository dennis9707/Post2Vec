python 02_process_to_tensor.py \
    --input_dir ../data/train \
    --out_dir ../data/albert/ \
    --model_type albert-base-v2 2>&1| tee ./albert.log