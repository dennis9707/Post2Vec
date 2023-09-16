CUDA_VISIBLE_DEVICES=2,1 python -u test_triplet_no_code.py \
    --data_dir ../../data/test_tensor \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_trinity_no_code1.log