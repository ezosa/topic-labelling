import pandas as pd
import numpy as np
import time
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier.train_utils import save_checkpoint
from classifier.CustomLoader import CustomLoader
from classifier.model import MLP

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='data/', type=str, help="")
argparser.add_argument('--train_data', default='train.csv', type=str, help="")
argparser.add_argument('--valid_data', default='valid.csv', type=str, help="")
argparser.add_argument('--label2id', default='iptc_tags2id.pkl', type=str, help="")
argparser.add_argument('--save_model_name', default='sbert-stt-2017', type=str, help="")
args = argparser.parse_args()

print("-"*10, "Train document classifier on SBERT embeddings", "-"*10)
print("data_path:", args.data_path)
print("train_data:", args.train_data)
print("valid_data:", args.valid_data)
print("label2id:", args.label2id)
print("save_model_name:", args.save_model_name)
print("-"*30 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

label2id = pickle.load(open(os.path.join(args.data_path, args.label2id), "rb"))
MODEL_NAME = args.save_model_name


def drop_col(c):
    try:
        unnamed = [x for x in c.columns if "nnamed" in x]
        c = c.drop(columns=unnamed)
    except KeyError:
        pass
    return c


def convert_codes(row):
    if pd.isna(row):
        return np.zeros(len(label2id))
    else:
        row = row.lower().split(', ')
        codes2id = np.array([label2id[label] for label in row if label in label2id])
        binary_label = np.zeros(len(label2id))
        binary_label[codes2id] = 1
    return binary_label


def read_dataset(dfname):
    now = time.time()
    print(f"[*] Reading {dfname}...")
    df = drop_col(pd.read_csv(dfname))
    X = torch.Tensor(df.drop(columns=["tags"]).to_numpy())
    y = torch.Tensor(df['tags'].apply(convert_codes))
    print(f"[!] Loaded {dfname}, took {time.time() - now:.2f}s")
    return X, y


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_path,
          criterion,
          num_epochs=20,
          eval_every_batches=100,
          best_valid_loss=float("Inf"),
          model_name="model"):

    # initialize running values
    now = time.time()
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    save_path = save_path if save_path.endswith("/") else save_path+"/"

    # training loop
    print("Start training for", num_epochs, "epochs...")
    model.to(device)
    model.float()
    model.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch + 1, "of", num_epochs)
        for train_batch in train_loader:
            labels = train_batch[1].to(device)
            content = train_batch[0].to(device)
            output = model(content).to(device)
            # labels & output shapes: [128, 103]

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every_batches == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for val_batch in valid_loader:
                        labels = val_batch[1].to(device)
                        content = val_batch[0].to(device)
                        output = model(content).to(device)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()


                # evaluation
                average_train_loss = running_loss / eval_every_batches
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                printline = 'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Time: {:.2f}s'\
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),\
                            average_train_loss, average_valid_loss, time.time() - now)
                print(printline)
                now = time.time()

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(save_path + model_name + '.pt', model, optimizer, best_valid_loss)
    print('Finished Training!')


if __name__ == "__main__":
    start = time.time()

    train_X, train_y = read_dataset(os.path.join(args.data_path, args.train_data))
    val_X, val_y = read_dataset(os.path.join(args.data_path, args.valid_data))

    print('train_X:', train_X.shape)
    print('train_y:', train_y.shape)
    EMBEDDING_DIM = 512
    OUTPUT_DIM = train_y.shape[1]
    BATCH_SIZE = 128

    model = MLP(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_custom_loader = CustomLoader(train_X, train_y)
    val_custom_loader = CustomLoader(val_X, val_y)

    train_loader = DataLoader(train_custom_loader, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_custom_loader, batch_size=BATCH_SIZE, shuffle=True)

    save_path = os.path.join(args.data_path, 'trained_models')
    print('save_path:', save_path)
    train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_path,
          criterion,
          num_epochs=30,
          model_name=MODEL_NAME)
    print(f"Execution time: {time.time() - start:.2f}s")
