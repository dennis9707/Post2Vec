python -u test_triplet.py \
    --test_batch_size 500 \
    --mlb_latest  \
    --model_type siamese 2>&1| tee ./logs/test_trinity.log