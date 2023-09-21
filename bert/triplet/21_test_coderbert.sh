CUDA_VISIBLE_DEVICES=3 python -u test_convert.py \
    --data_dir ../../data/test_codebert-tail \
    --test_batch_size 250 \
    --code_bert microsoft/codebert-base \
    --mlb_latest 2>&1| tee ./tail_logs/test_codebert.log