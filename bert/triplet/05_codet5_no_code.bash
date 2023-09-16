CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --nproc_per_node=2 train_no_code.py \
    --data_folder ../../data/codet5 \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --code_bert Salesforce/codet5-base \
    --model_path ../../data/results/Salesforce/codet5-base_01-16-08-48-16_code/final-epoch-0 \
    --remove_component code \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_codet5_no_code0916.log


