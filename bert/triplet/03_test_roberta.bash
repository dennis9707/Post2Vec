CUDA_VISIBLE_DEVICES=2,3 python -u test_triplet.py \
    --data_dir ../../data/test_roberta \
    --test_batch_size 500 \
    --code_bert roberta-base \
    --mlb_latest 2>&1| tee ./logs/test_roberta_tensor.log