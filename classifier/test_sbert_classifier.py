import pickle
import torch
import time
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score

from classifier.train_utils import load_checkpoint
from classifier.CustomLoader import CustomLoader
from classifier.model import MLP


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='', type=str, help="")
argparser.add_argument('--test_csv', default='', type=str, help="")
argparser.add_argument('--label2id', default='iptc_tags2id.pkl', type=str, help="")
argparser.add_argument('--model_name', default='sbert-stt-2017', type=str, help="")
args = argparser.parse_args()

print("\n" + "-"*10, "Test SBERT classifier", "-"*10)
print("data_path:", args.data_path)
print("test_csv:", args.test_csv)
print("label2id:", args.label2id)
print("model_name:", args.model_name)
print("-"*50 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

label2id = pickle.load(open(os.path.join(args.data_path, args.label2id), "rb"))
MODEL_NAME = args.model_name


def evaluate(model, test_loader, threshold=0.05):
    y_pred = []
    y_true = []
    print('threshold:', threshold)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            
            labels = test_batch[1].to(device)
            content = test_batch[0].to(device)
            
            y_score_i = model(content).to(device)
            y_pred_i = (y_score_i > threshold).int()

            y_pred.extend(y_pred_i.tolist())
            y_true.extend(labels.tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    results = {'f1': macro_f1,
               'recall': recall,
               'precision': precision}

    return results


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


if __name__ == "__main__":

    start = time.time()

    test_X, test_y = read_dataset(os.path.join(args.data_path, args.test_csv))
    EMBEDDING_DIM = 512
    OUTPUT_DIM = test_y.shape[1]
    BATCH_SIZE = 128

    model = MLP(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    test_loader = CustomLoader(test_X, test_y)

    train_loader = DataLoader(test_loader, batch_size=BATCH_SIZE, shuffle=False)

    model_path = os.path.join(args.data_path, 'trained_models')
    load_checkpoint(os.path.join(model_path, args.model_name + '.pt'), model, optimizer, device)
    results = evaluate(model, test_loader)
    print(results)
