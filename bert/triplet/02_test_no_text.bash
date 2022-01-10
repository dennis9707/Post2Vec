CUDA_VISIBLE_DEVICES=4,5 python -u test_triplet_no_text.py \
    --data_dir ../../data/test_tensor \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_trinity_no_text.log