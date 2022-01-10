CUDA_VISIBLE_DEVICES=0,1 python -u test_triplet.py \
    --data_dir ../../data/test_bertoverflow \
    --test_batch_size 400 \
    --code_bert jeniya/BERTOverflow \
    --mlb_latest 2>&1| tee ./logs/test_bertoverflow_tensor.log