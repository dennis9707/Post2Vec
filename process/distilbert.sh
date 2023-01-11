python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/distilroberta/ \
    --model_type distilroberta-base 2>&1| tee ./distilroberta.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/distilbert/ \
    --model_type distilbert-base-uncased 2>&1| tee ./distilbert.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/codebert-small/ \
    --model_type huggingface/CodeBERTa-small-v1 2>&1| tee ./codebert-small.log
