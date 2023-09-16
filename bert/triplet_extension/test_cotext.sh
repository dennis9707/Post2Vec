CUDA_VISIBLE_DEVICES=1,2 python -u test_triplet.py \
    --data_dir ../../data/cotext-test \
    --code_bert razent/cotext-2-cc \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test-cotext-csv.log