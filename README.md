# Post2Vec Extension

## Data

https://zenodo.org/record/5604548#.YXoG7NZBw1I

## Post2Vec Extension

- Train File: ./bert/triplet/train_trinity.bash
- Test File: ./bert/triplet/test_triplet.py

```python
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train_trinity.py \
    --data_folder ../../data/train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 2 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5  2>&1| tee train_trinity.log
```

distributed training

```python
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train_trinity.py \
    --data_folder ../../data/train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 2 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5  2>&1| tee train_trinity.log
```
