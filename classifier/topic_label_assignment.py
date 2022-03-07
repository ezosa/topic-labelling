import pickle
import torch
import time
import os
import numpy as np
import pandas as pd
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier.train_utils import load_checkpoint
from classifier.CustomLoader import CustomLoader
from classifier.model import MLP

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', default='', type=str, help="")
argparser.add_argument('--sbert_csv', default='', type=str, help="")
argparser.add_argument('--topic_csv', default='', type=str, help="")
argparser.add_argument('--label2id', default='iptc_tags2id.pkl', type=str, help="")
argparser.add_argument('--model_name', default='sbert-stt-2017', type=str, help="")
argparser.add_argument('--num_docs', default=20, type=int, help="")
argparser.add_argument('--num_labels', default=5, type=int, help="")
argparser.add_argument('--output_path', default='', type=str, help="")
argparser.add_argument('--propagate', default=0, type=int, help="")
argparser.add_argument('--lang', default='fi', type=str, help="")
args = argparser.parse_args()

print("\n" + "-" * 10, "Assign topic labels using SBERT classifier", "-" * 10)
print("data_path:", args.data_path)
print("sbert_csv:", args.sbert_csv)
print("topic_csv:", args.topic_csv)
print("label2id:", args.label2id)
print("model_name:", args.model_name)
print("num_docs:", args.num_docs)
print("num_labels:", args.num_labels)
print("output_path:", args.output_path)
print("propagate:", args.propagate)
print("lang:", args.lang)
print("-" * 50 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

label2id = pickle.load(open(os.path.join(args.data_path, args.label2id), "rb"))
id2label = {label2id[label]:label for label in label2id}
iptc_data = json.load(open(os.path.join(args.data_path, "IPTC_parsed_STT.json"), "r"))


def classify(model, test_loader, threshold=0.03):
    pred = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            content = test_batch[0].to(device)
            score_batch = model(content).to(device)
            pred_batch = (score_batch > threshold).int()
            pred.extend(pred_batch.tolist())
    pred = np.array(pred, dtype=int)
    return pred


def convert_pred_to_labels(pred):
    pred_labels = []
    for row_index in range(pred.shape[0]):
        row = pred[row_index]
        pred_ids = np.nonzero(row)[0]
        if len(pred_ids) > 0:
            node_labels = [id2label[pred_id] for pred_id in pred_ids if pred_id in id2label]
            pred_labels.extend(node_labels)
            if args.propagate == 1:
                node_parents = []
                for label in node_labels:
                    if label in iptc_data:
                        parent_name = iptc_data[label]['parent_name']
                        while parent_name is not None:
                            node_parents.append(parent_name)
                            if parent_name in iptc_data:
                                parent_name = iptc_data[parent_name]['parent_name']
                            else:
                                parent_name = None
                pred_labels.extend(node_parents)
    # pred_labels = [p for sublist in pred_labels for p in sublist]
    # pred_labels = list(set(pred_labels))
    #print("pred_labels:", pred_labels)
    return pred_labels



def convert_pred_to_labels2(pred):
    pred_labels = []
    for row_index in range(pred.shape[0]):
        row_labels = []
        row = pred[row_index]
        pred_ids = np.nonzero(row)[0]
        if len(pred_ids) > 0:
            node_labels = [id2label[pred_id] for pred_id in pred_ids if pred_id in id2label]
            row_labels.extend(node_labels)
            if args.propagate == 1:
                node_parents = []
                for label in node_labels:
                    if label in iptc_data:
                        parent_name = iptc_data[label]['parent_name']
                        while parent_name is not None:
                            node_parents.append(parent_name)
                            if parent_name in iptc_data:
                                parent_name = iptc_data[parent_name]['parent_name']
                            else:
                                parent_name = None
                row_labels.extend(node_parents)
        pred_labels.append(row_labels)
    # pred_labels = [p for sublist in pred_labels for p in sublist]
    # pred_labels = list(set(pred_labels))
    #print("pred_labels:", pred_labels)
    return pred_labels


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


def read_unlabelled_dataset(dfname):
    now = time.time()
    print(f"[*] Reading {dfname}...")
    df = drop_col(pd.read_csv(dfname))
    if 'tags' in df.columns:
        X = torch.Tensor(df.drop(columns=["tags"]).to_numpy())
    else:
        X = torch.Tensor(df.to_numpy())
    print(f"[!] Loaded {dfname}, took {time.time() - now:.2f}s")
    return X


if __name__ == "__main__":
    start = time.time()

    # hyperparams
    EMBEDDING_DIM = 512
    OUTPUT_DIM = 1274 #len(label2id)
    BATCH_SIZE = args.num_docs

    # initialize model
    model = MLP(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    model_path = os.path.join(args.data_path, 'trained_models')
    load_checkpoint(os.path.join(model_path, args.model_name + '.pt'), model, optimizer, device)

    sbert_emb = read_unlabelled_dataset(os.path.join(args.data_path, args.sbert_csv))
    print('sbert_emb:', sbert_emb.shape)

    output_file = open(os.path.join(args.output_path, args.model_name + '.txt'), 'w', encoding='utf-8')

    # create data loader
    dummy_labels = torch.zeros((sbert_emb.shape[0], OUTPUT_DIM))
    test_loader = CustomLoader(sbert_emb, dummy_labels)
    test_loader = DataLoader(test_loader, batch_size=30, shuffle=False)
    # ontology mapping and label propagation
    pred_ids = classify(model, test_loader)
    pred_labels = convert_pred_to_labels2(pred_ids)
    for k in range(len(pred_labels)):
        print("Topic", k+1, ":")
        label_counts = Counter(pred_labels[k])
        print('unique labels:', len(label_counts))
        sorted_labels = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
        print('sorted_labels:', sorted_labels)
        if len(sorted_labels) > args.num_labels:
            sorted_labels = sorted_labels[:args.num_labels]
        final_labels = [sorted_labels[i][0] for i in range(len(sorted_labels))]
        final_labels = ', '.join(final_labels)
        print("final_labels:", final_labels)
        output_file.write(final_labels + '\n')

    output_file.close()
    print("Done! Saved final labels to", args.output_path, "!")


