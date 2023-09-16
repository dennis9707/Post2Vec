CUDA_VISIBLE_DEVICES=6,7 python -u test_triplet.py \
    --data_dir ../../data/codebert-small-test \
    --code_bert huggingface/CodeBERTa-small-v1 \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-codebert-small.log