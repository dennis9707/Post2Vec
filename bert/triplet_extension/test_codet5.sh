CUDA_VISIBLE_DEVICES=5,6 python -u test_triplet.py \
    --data_dir ../../data/codet5-base-test \
    --code_bert Salesforce/codet5-base \
    --test_batch_size 250 \
    --mlb_latest 2>&1| tee ./logs/test-codet5-base-demo.log
