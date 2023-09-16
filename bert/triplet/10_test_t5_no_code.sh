CUDA_VISIBLE_DEVICES=5,2 python -u test_triplet_no_code.py \
    --data_dir ../../data/codet5-base-test \
    --code_bert Salesforce/codet5-base \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_codet5_no_code-checkpoint499.log