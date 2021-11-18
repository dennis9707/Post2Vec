from sklearn import metrics
import pandas as pd
from data_structure.question import DistilBERTQuestionDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing
from model.model import RobertaForMultiLable
from model.loss import loss_fn
import gc
import numpy as np
from transformers import AutoTokenizer
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from util.util import save_ckp, load_ckp
import argparse
import os
import random
from datetime import datetime

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(input_train, input_valid, mlb, args):
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    valid_loss_min = np.Inf
    checkpoint_path = './results/checkpoint/current_checkpoint_roberta_ts1000.pt'
    best_model_path = './results/best_model/best_model_roberta_ts1000.pt'
    train = pd.read_pickle(args.train_data_file)
    valid = pd.read_pickle(args.valid_data_file)
    # hyperparameters
    TRAIN_BATCH_SIZE = args.train_batch_size
    VALID_BATCH_SIZE = args.valid_batch_size
    EPOCHS = args.epoch
    LEARNING_RATE = args.learning_rate

    training_set = DistilBERTQuestionDataset(train, mlb, tokenizer)
    valid_set = DistilBERTQuestionDataset(valid, mlb, tokenizer)

    train_data_loader = DataLoader(training_set,
                                   batch_size=TRAIN_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0
                                   )

    valid_data_loader = DataLoader(valid_set,
                                   batch_size=VALID_BATCH_SIZE,
                                   num_workers=0
                                   )

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_train_steps = int(len(training_set) / TRAIN_BATCH_SIZE * 10)
    model = RobertaForMultiLable(170)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    print(f'number of training steps {n_train_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=n_train_steps)
    gc.collect()
    torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        model.train()
        for batch_idx, data in enumerate(train_data_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['labels'].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(
                input_ids=ids, attention_mask=mask)

            loss = loss_fn(outputs, targets)
            if batch_idx % 100 == 0:
                print(
                    f'Epoch: {epoch}, Batch: {batch_idx}， Loss:  {loss.item()}')
                current_time = datetime.now().strftime("%H:%M:%S")
                print("Current Time =", current_time)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('############# Epoch {}: Training End     #############'.format(epoch))
        print('############# Epoch {}: Validation Start   #############'.format(epoch))
        ######################
        # validate the model #
        ######################

        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_data_loader, 0):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['labels'].to(device, dtype=torch.float)
                outputs = model(
                    input_ids=ids, attention_mask=mask)

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(
                    outputs).cpu().detach().numpy().tolist())
                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + \
                    ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            test(fin_outputs, fin_targets)
            print(
                '############# Epoch {}: Validation End     #############'.format(epoch))
            # calculate average losses
            #print('before cal avg train loss', train_loss)
            train_loss = train_loss/len(train_data_loader)
            valid_loss = valid_loss/len(valid_data_loader)
            # print training/validation statistics
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            # TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

        print('############# Epoch {}  Done   #############\n'.format(epoch))
    return model


def test(outputs, targets):
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    recall_score_micro = metrics.recall_score(
        targets, outputs, average='micro')
    recall_score_macro = metrics.recall_score(
        targets, outputs, average='macro')
    precision_score_micro = metrics.precision_score(
        targets, outputs, average='micro')
    precision_score_macro = metrics.precision_score(
        targets, outputs, average='macro')
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"Recall Score (Micro) = {recall_score_micro}")
    print(f"Recall Score (Macro) = {recall_score_macro}")
    print(f"Precision Score (Micro) = {precision_score_micro}")
    print(f"Precision Score (Macro) = {precision_score_macro}")


def main():
    ############################ model arguments settings ############################
    parser = argparse.ArgumentParser(
        description='Multi-label Classifier based on Bert-based Models')

    # Required parameters
    parser.add_argument("--train_data_file", default="../data/questions/Train_Questions54TS1000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--valid_data_file", default="../data/questions/Valid_Questions54TS1000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="The testing data file")

    parser.add_argument("--vocab_file", default="../data/tags/20211110/ts1000/_2_1_commonTags.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--epoch", default=3, type=int,
                        help="The number of epoch")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('-dropout', type=float, default=0.1,
                        help='the probability for dropout [default: 0.1]')
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--metric_threshold', type=float, default=0.5,
                        help="threshold to metric calculation")

    args = parser.parse_args()

    # Build Tag Vocab
    seed_everything(args.seed)
    tab_vocab_path = args.vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    args.mlb = mlb
    input_train = args.train_data_file
    input_valid = args.valid_data_file
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    train(input_train, input_valid, mlb, args)


if __name__ == '__main__':
    main()
