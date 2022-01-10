CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet.py \
    --data_dir ../../data/test_albert \
    --test_batch_size 500 \
    --code_bert albert-base-v2 \
    --mlb_latest 2>&1| tee ./logs/test_albert_tensor.log