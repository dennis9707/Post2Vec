CUDA_VISIBLE_DEVICES=3,4 python -u test_triplet_csv.py \
    --data_dir ../../data/test \
    --test_batch_size 500 \
    --mlb_latest 2>&1| tee ./logs/test_codebert_to_csv.log


docker run --name=junda-ptm4tag --gpus all -it -v /mnt/hdd1/jundahe/post2vec:/usr/src nvcr.io/nvidia/pytorch:21.05-py3 /bin/bash