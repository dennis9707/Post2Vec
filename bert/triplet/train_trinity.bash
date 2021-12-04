CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_trinity.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 7e-5  2>&1| tee train_trinity.log
