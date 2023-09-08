CUDA_VISIBLE_DEVICES=1 python train_title.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 256 \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --include_component title \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_codebert_title-fpO2.log
