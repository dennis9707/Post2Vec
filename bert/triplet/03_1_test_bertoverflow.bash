CUDA_VISIBLE_DEVICES=3,4 python -u test_triplet.py \
    --data_dir ../../data/test_bertoverflow \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_trinity_tensor.log