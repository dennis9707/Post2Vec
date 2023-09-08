CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/plbart-test \
    --code_bert uclanlp/plbart-base \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-plbart.log
