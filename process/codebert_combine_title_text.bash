python 02_1_process_title_descp.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/combine/ \
    --model_type microsoft/codebert-base 2>&1| tee ./combine.log