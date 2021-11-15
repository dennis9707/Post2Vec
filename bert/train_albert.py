from sklearn import metrics
import pandas as pd
from data_structure.question import QuestionDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing
from model.model import CodeBERTModel
from model.loss import loss_fn
import gc
import numpy as np
from transformers import AutoTokenizer
from util.util import save_ckp, load_ckp
import os


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


def train(input_train, input_valid, mlb):
    # device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    valid_loss_min = np.Inf
    checkpoint_path = './results/checkpoint/current_checkpoint.pt'
    best_model = './results/best_model/best_model.pt'
    train = pd.read_pickle(input_train)
    valid = pd.read_pickle(input_valid)
    # hyperparameters
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 1e-05

    training_set = QuestionDataset(train, mlb, tokenizer)
    valid_set = QuestionDataset(valid, mlb, tokenizer)

    train_data_loader = DataLoader(training_set,
                                   batch_size=TRAIN_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2
                                   )

    valid_data_loader = DataLoader(valid_set,
                                   batch_size=VALID_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2
                                   )

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CodeBERTModel(9)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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

            outputs = model(ids, mask)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if batch_idx % 5000 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('############# Epoch {}: Training End     #############'.format(epoch))

        print('############# Epoch {}: Validation Start   #############'.format(epoch))
        ######################
        # validate the model #
        ######################

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(valid_data_loader, 0):
                ids = data['input_ids'].to(device, dtype=torch.long)
                targets = data['labels'].to(device, dtype=torch.float)
                print("here")
                print()
                outputs = model(ids)
                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + \
                    ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

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


def test():
    outputs, targets = validation(epoch, test_data_loader)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")


def main():
    ############################ model arguments settings ############################
    # parser = argparse.ArgumentParser(
    #     description='Multi-label Classifier based on Multi-component')
    # basic settings
    # parser.add_argument('-batch-size', type=int, default=32,
    #                     help='batch size for training [default: 32]')
    # parser.add_argument('-epochs', type=int, default=5,
    #                     help='number of epochs for train [default: 5]')

    # ############################# tuned parameter #############################
    # parser.add_argument('-lr', type=float, default=0.0001,
    #                     help='initial learning rate [default: 0.001]')
    # parser.add_argument('-dropout', type=float, default=0.3,
    #                     help='the probability for dropout [default: 0.3]')
    # parser.add_argument('-hidden-dim', type=int, default=512,
    #                     help='number of hidden dimension of fully connected layer [default: 512]')
    ############################################################################

    # Build Tag Vocab
    seed_everything(1234)
    tab_vocab_path = "../data/tags/20211110/top10/common.csv"
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])

    input_train = "../data/questions/Train_Questions54Top10.pkl"
    input_valid = "../data/questions/Valid_Questions54Top10.pkl"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    train(input_train, input_valid, mlb)


if __name__ == '__main__':
    main()
