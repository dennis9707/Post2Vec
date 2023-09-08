# Post2Vec Extension

## Data

https://zenodo.org/record/5604548#.YXoG7NZBw1I

## Download Data

### Train

```shell
    wget https://zenodo.org/record/5604548/files/train.tar.gz
```

### Test

```shell
    wget https://zenodo.org/record/5604548/files/test.tar.gz
```
### Tag

```shell
    wget https://zenodo.org/record/5604548/files/_1_commonTags.csv
```

### Unzip
```
   tar -xvf file.tar.gz 
```

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



Model Name:
```
bert-base-uncased
albert-base-v2
jeniya/BERTOverflow
microsoft/codebert-base
Salesforce/codet5-base
huggingface/CodeBERTa-small-v1
distilbert-base-uncased
distilroberta-base
razent/cotext-2-cc
Salesforce/codet5-small
uclanlp/plbart-base
roberta-base
```
