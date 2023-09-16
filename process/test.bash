python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_albert/ \
    --model_type albert-base-v2 2>&1| tee ./albert-test.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/test_bertoverflow \
    --model_type jeniya/BERTOverflow 2>&1| tee ./bertoverflow-test.log

# python 02_process_to_tensor.py \
#     --input_dir ../data/processed_test \
#     --out_dir ../data/codet5-small-test/ \
#     --model_type Salesforce/codet5-small 2>&1| tee ./codet5-small-test.log

# python 02_process_to_tensor.py \
#     --input_dir ../data/processed_test \
#     --out_dir ../data/distilroberta-test/ \
#     --model_type distilroberta-base 2>&1| tee ./distilroberta-test.log

# python 02_process_to_tensor.py \
#     --input_dir ../data/processed_test \
#     --out_dir ../data/distilbert-test/ \
#     --model_type distilbert-base-uncased 2>&1| tee ./distilbert-test.log

# python 02_process_to_tensor.py \
#     --input_dir ../data/processed_test \
#     --out_dir ../data/codebert-small/ \
#     --model_type huggingface/CodeBERTa-small-v1 2>&1| tee ./codebert-small.log
