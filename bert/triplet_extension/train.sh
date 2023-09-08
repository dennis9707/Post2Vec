CUDA_VISIBLE_DEVICES=0 python train_triplet.py \
    --data_folder ../../data/plbart \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 32 \
    --logging_steps 100 \
    --code_bert uclanlp/plbart-base \
    --model_path ../../data/results/uclanlp/plbart-base_01-13-07-06-54_/epoch-0-file-492/t_bert.pt \
    --num_train_epochs 1 \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee train_plbart-test.log