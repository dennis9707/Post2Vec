CUDA_VISIBLE_DEVICES=3,0 python -u test_triplet_no_text.py \
    --data_dir ../../data/codet5-base-test \
    --code_bert Salesforce/codet5-base \
    --test_batch_size 400 \
    --mlb_latest 2>&1| tee ./logs/test_codet5_no_text.log