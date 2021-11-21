CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train_trinity.py \
    --data_folder ../../data/train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --logging_steps 100 \
    --save_steps 1000 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 8e-5  2>&1| tee train_trinity.log
