CUDA_VISIBLE_DEVICES=0,1 python -u test_triplet.py \
    --data_dir ../../data/test_tensor \
    --test_batch_size 500 \
    --code_bert microsoft/codebert-base \
    --mlb_latest 2>&1| tee ./logs/test_trinity_tensor.log