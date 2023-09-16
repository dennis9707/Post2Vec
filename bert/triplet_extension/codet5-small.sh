CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch \
    --nproc_per_node=2 train_triplet.py \
    --data_folder ../../data/codet5-small-test \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --code_bert Salesforce/codet5-small \
    --num_train_epochs 1 \
    --fp16 \
    --fp16_opt_level O2 \
    --learning_rate 7e-5  2>&1| tee train_codet5-small.log