python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/cotext/ \
    --model_type razent/cotext-2-cc 2>&1| tee ./cotext.log


python 02_process_to_tensor.py \
    --input_dir ../data/processed_test \
    --out_dir ../data/cotext-test/ \
    --model_type razent/cotext-2-cc 2>&1| tee ./cotext-test.log