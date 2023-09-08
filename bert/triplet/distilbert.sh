CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
    --nproc_per_node=2 train_triplet.py \
    --data_folder ../../data/distilbert \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --code_bert distilbert-base-uncased \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee train_distilbert.log