CUDA_VISIBLE_DEVICES=0,1 python -u test_triplet_csv.py \
    --data_dir ../../data/test \
    --test_batch_size 500 \
    --code_bert bert-base-uncased \
    --mlb_latest 2>&1| tee ./logs/test_bert_to_csv.log