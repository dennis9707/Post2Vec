CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 train_no_code.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --code_bert albert-base-v2 \ 
    --no_code \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_triplet_albert_no_code-fpO2.log
