python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_tensor/ \
    --model_type microsoft/codebert-base 2>&1| tee ./test_tensor.log