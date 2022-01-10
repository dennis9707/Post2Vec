python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_bert/ \
    --model_type bert-base-uncased 2>&1| tee ./test_bert.log