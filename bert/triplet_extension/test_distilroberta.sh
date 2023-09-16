CUDA_VISIBLE_DEVICES=2,5 python -u test_triplet.py \
    --data_dir ../../data/distilbert-test \
    --code_bert distilbert-base-uncased \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./test-distilbert1.log

# CUDA_VISIBLE_DEVICES=1,5 python -u test_triplet.py \
#     --data_dir ../../data/distilroberta-test \
#     --code_bert distilroberta-base \
#     --test_batch_size 500 \
#     --mlb_latest 2>&1| tee ./test-distilroberta.log
# CUDA_VISIBLE_DEVICES=2,3 python -u test_triplet.py \
#     --data_dir ../../data/codebert-small-test \
#     --code_bert huggingface/CodeBERTa-small-v1 \
#     --test_batch_size 500 \
#     --mlb_latest 2>&1| tee ./test-codebert-small.log