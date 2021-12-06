CUDA_VISIBLE_DEVICES=2,3 python -u -m torch.distributed.launch \
    --nproc_per_node=2  \
    train_siamese.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --train_batch_size 6 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --fp16   \
    --learning_rate 7e-5  2>&1| tee train_siamese.log 