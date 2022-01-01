python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test/roberta/ \
    --model_type roberta-base 2>&1| tee ./roberta.log