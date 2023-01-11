python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/codet5-small-test/ \
    --model_type Salesforce/codet5-small 2>&1| tee ./test_codet5-small.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/codet5-base-test/ \
    --model_type Salesforce/codet5-base 2>&1| tee ./test_codet5-base.log
    
python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/distilroberta-test/ \
    --model_type distilroberta-base 2>&1| tee ./test_distilroberta.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/distilbert-test/ \
    --model_type distilbert-base-uncased 2>&1| tee ./test_distilbert.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/codebert-small-test/ \
    --model_type huggingface/CodeBERTa-small-v1 2>&1| tee ./test_codebert-small.log

