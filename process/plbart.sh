python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/plbart/ \
    --model_type uclanlp/plbart-base 2>&1| tee ./plbart.log
python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/plbart-test/ \
    --model_type uclanlp/plbart-base 2>&1| tee ./plbart-test.log