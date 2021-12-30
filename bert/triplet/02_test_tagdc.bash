CUDA_VISIBLE_DEVICES=2,3 python -u test_tensor_data.py \
    --data_dir ../../data/tagdc_test_tensor \
    --vocab_file ../../data/tagdc_csv/tagdc_commonTags.csv \
    --test_batch_size 200 \
    --mlb_latest  \
    --load_tensor 2>&1| tee ./logs/test_tagdc.log