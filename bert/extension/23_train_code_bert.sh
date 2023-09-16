CUDA_VISIBLE_DEVICES=2 python train_title.py \
    --data_folder ../../data/tensor_data \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 64 \
    --logging_steps 100 \
    --num_train_epochs 2 \
    --include_component title \
    --code_bert bert-base-uncased \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee ./train_logs/train_bert_code-fpO2.log