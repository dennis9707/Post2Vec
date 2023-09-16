CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
    --nproc_per_node=2 train_no_title.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 8 \
    --logging_steps 100 \
    --gradient_accumulation_steps 1 \
    --model_path ../../data/results/microsoft/codebert-base_09-13-16-12-52_title/epoch-0-file-19/t_bert.pt \
    --num_train_epochs 1 \
    --code_bert microsoft/codebert-base \
    --remove_component title \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_codebert_no_title0917.log
