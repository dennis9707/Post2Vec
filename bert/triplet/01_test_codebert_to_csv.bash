CUDA_VISIBLE_DEVICES=3,4 python -u test_triplet_csv.py \
    --data_dir ../../data/test \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_codebert_to_csv.log