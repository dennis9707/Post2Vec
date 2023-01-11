python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/codet5-small/ \
    --model_type Salesforce/codet5-small 2>&1| tee ./codet5-small.log

python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/codet5-base/ \
    --model_type Salesforce/codet5-base 2>&1| tee ./codet5-base.log
