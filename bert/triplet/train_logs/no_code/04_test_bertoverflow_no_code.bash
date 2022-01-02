CUDA_VISIBLE_DEVICES=3,4 python -u test_triplet_no_code.py \
    --data_dir ../../data/test_bertoverflow \
    --test_batch_size 500 \
    --no_code \
    --mlb_latest 2>&1| tee ./logs/test_bertoverflow_tensor_no_code.log