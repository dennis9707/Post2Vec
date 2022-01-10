python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/bert/ \
    --model_type bert-base-uncased 2>&1| tee ./bert.log