CUDA_VISIBLE_DEVICES=3,1 python -u convert_checkpoint.py \
    --code_bert microsoft/codebert-base \
    --mlb_latest 2>&1| tee ./convert.log