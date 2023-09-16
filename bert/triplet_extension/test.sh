CUDA_VISIBLE_DEVICES=3,7 python -u test_triplet.py \
    --data_dir ../../data/codet5-small-test \
    --code_bert Salesforce/codet5-small \
    --test_batch_size 400 \
    --mlb_latest 2>&1| tee ./test-codet5-small.log

# CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet.py \
#     --data_dir ../../data/distilbert-test \
#     --code_bert distilbert-base-uncased \
#     --test_batch_size 1 \
#     --mlb_latest 2>&1| tee ./logs/test-distilbert-1.log

# CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet.py \
#     --data_dir ../../data/distilroberta-test \
#     --code_bert distilroberta-base \
#     --test_batch_size 1 \
#     --mlb_latest 2>&1| tee ./logs/test-distilroberta-1.log

# CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet.py \
#     --data_dir ../../data/codebert-small-test \
#     --code_bert huggingface/CodeBERTa-small-v1 \
#     --test_batch_size 1 \
#     --mlb_latest 2>&1| tee ./logs/test-codebert-small-1.log

