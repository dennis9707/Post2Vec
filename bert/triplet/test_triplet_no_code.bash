CUDA_VISIBLE_DEVICES=3,4 python -u test_triplet.py \
    --test_batch_size 500 \
    --no_code \
    --mlb_latest 2>&1| tee ./logs/test_trinity.log