CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train_trinity.py \
    --data_folder ../../data/train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --logging_steps 10 \
    --save_steps 10000 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 4e-5  2>&1| tee train_trinity.log
