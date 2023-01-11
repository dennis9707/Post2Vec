CUDA_VISIBLE_DEVICES=2,5 python -u test_triplet_csv.py \
    --data_dir ../../data/test \
    --test_batch_size 500 \
    --code_bert roberta-base \
    --mlb_latest 2>&1| tee ./logs/test_robert_to_csv.log