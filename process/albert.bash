python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/albert/ \
    --model_type albert-base-v2 2>&1| tee ./albert.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_albert/ \
    --model_type albert-base-v2 2>&1| tee ./test_albert.log