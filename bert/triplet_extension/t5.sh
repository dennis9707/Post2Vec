CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch \
    --nproc_per_node=2 train_triplet.py \
    --data_folder ../../data/t5-train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 4 \
    --logging_steps 100 \
    --gradient_accumulation_steps 8 \
    --code_bert t5-base \
    --num_train_epochs 1 \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee train_t5-base.log

# CUDA_VISIBLE_DEVICES=5,7 python -u test_triplet.py \
#     --data_dir ../../data/t5-test \
#     --code_bert t5-base \
#     --test_batch_size 500 \
#     --mlb_latest 2>&1| tee ./logs/test-t5-base.log