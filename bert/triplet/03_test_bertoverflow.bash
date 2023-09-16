CUDA_VISIBLE_DEVICES=6,7 python -u test_triplet_csv.py \
    --data_dir ../../data/test_bertoverflow \
    --test_batch_size 400 \
    --code_bert jeniya/BERTOverflow \
    --mlb_latest 2>&1| tee ./logs/test_bertoverflow_tensor.log