CUDA_VISIBLE_DEVICES=2,3 python -u test_triplet_no_title.py \
    --data_dir ../../data/codet5-base-test \
    --test_batch_size 500 \
    --code_bert Salesforce/codet5-base \
    --mlb_latest 2>&1| tee ./logs/test_codet5_no_title.log