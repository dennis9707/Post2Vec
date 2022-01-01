python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_bertoverflow/ \
    --model_type jeniya/BERTOverflow 2>&1| tee ./test_bertoverflow.log