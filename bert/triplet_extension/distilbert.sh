CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
    --nproc_per_node=2 train_triplet.py \
    --data_folder ../../data/distilbert-train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --code_bert distilbert-base-uncased \
    --num_train_epochs 1 \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee train_distilbert-0913.log

# CUDA_VISIBLE_DEVICES=2,3 python -u test_triplet.py \
#     --data_dir ../../data/bart-test \
#     --code_bert facebook/bart-base \
#     --test_batch_size 500 \
#     --mlb_latest 2>&1| tee ./logs/test-bart-base.log