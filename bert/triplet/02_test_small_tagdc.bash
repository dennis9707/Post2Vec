CUDA_VISIBLE_DEVICES=2,3 python -u test_tensor_data.py \
    --data_dir ../../data/small_tagdc/final_data/test \
    --vocab_file ../../data/small_tagdc/small_tagdc_commonTags.csv \
    --test_batch_size 200 \
    --mlb_latest  \
    --load_tensor 2>&1| tee ./logs/test_small_tagdc.log