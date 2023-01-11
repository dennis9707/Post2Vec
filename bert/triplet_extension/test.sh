CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/distilbert-test \
    --code_bert distilbert-base-uncased \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-distilbert.log

CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/distilroberta-test \
    --code_bert distilroberta-base \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-distilroberta.log

CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/codebert-small-test \
    --code_bert huggingface/CodeBERTa-small-v1 \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-codebert-small.log

CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/codet5-small-test \
    --code_bert Salesforce/codet5-small \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-codet5-small.log
