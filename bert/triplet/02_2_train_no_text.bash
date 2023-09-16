CUDA_VISIBLE_DEVICES=5,3 python -m torch.distributed.launch \
    --nproc_per_node=2 train_no_text.py \
    --data_folder ../../data/codet5-base \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --code_bert microsoft/codebert-base \
    --remove_component text \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_codebert_no_text0916.log
