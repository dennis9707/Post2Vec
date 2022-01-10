CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet.py \
    --data_dir ../../data/test_bert \
    --test_batch_size 500 \
    --code_bert bert-base-uncased \
    --mlb_latest 2>&1| tee ./logs/test_bert_tensor.log