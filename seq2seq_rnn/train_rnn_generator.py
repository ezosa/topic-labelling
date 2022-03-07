import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

import numpy as np
import random
import math
import time
import io
import os
from collections import Counter

from seq2seq_rnn.Seq2Seq import Seq2Seq, Encoder, Decoder, Attention
from seq2seq_rnn.rnn_utils import load_pretrained_embeddings

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='rnn_stt', type=str, help="model name")
argparser.add_argument('--save_path', default='trained_models/', type=str, help="save directory")
argparser.add_argument('--train_data_path', default='data/', type=str, help="path to training data")
argparser.add_argument('--top_terms_strategy', default='tfidf', type=str, help="tfidf or sent")
argparser.add_argument('--pretrained_emb', default=None, type=str, help="path to pretrained embeddings")
args = argparser.parse_args()

print("\n"+"-"*10, "Train RNN seq2seq for Label Generation", "-"*10)
print("model_name:", args.model_name)
print("save_path:", args.save_path)
print("train_data_path:", args.train_data_path)
print("top_terms_strategy:", args.top_terms_strategy)
print("pretrained_emb:", args.pretrained_emb)
print("-"*70 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = get_tokenizer('basic_english')


def generate_batch(data_batch):
    src_batch = []
    trg_batch = []
    for (src_item, trg_item) in data_batch:
        src_batch.append(torch.cat([torch.tensor([SOS_IDX]), src_item, torch.tensor([EOS_IDX])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([SOS_IDX]), trg_item, torch.tensor([EOS_IDX])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    src_batch = src_batch.T
    trg_batch = trg_batch.T
    # print("src_batch:", src_batch.shape)
    # print("trg_batch:", trg_batch.shape)
    return src_batch, trg_batch


def build_vocab(filepath, tokenizer, freq=2):
    counter = Counter()
    lines = open(filepath, 'r', encoding="utf8").readlines()
    for line in lines:
        counter.update(tokenizer(line.lower()))
    return Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=freq)


def build_combined_vocab(filepaths, tokenizer, freq=3):
    print("Building vocab with min_freq =", freq)
    counter = Counter()
    for filepath in filepaths:
        lines = open(filepath, 'r', encoding="utf8").readlines()
        for line in lines:
            counter.update(tokenizer(line.lower()))
    return Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=freq)


def data_process(src_filepath, trg_filepath):
    raw_src_iter = iter(io.open(src_filepath, encoding="utf8"))
    raw_trg_iter = iter(io.open(trg_filepath, encoding="utf8"))
    data = []
    for (raw_src, raw_trg) in zip(raw_src_iter, raw_trg_iter):
      src_tensor = torch.tensor([combined_vocab[token] for token in tokenizer(raw_src)], dtype=torch.long)
      trg_tensor = torch.tensor([combined_vocab[token] for token in tokenizer(raw_trg)], dtype=torch.long)
      data.append((src_tensor, trg_tensor))
    return data


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        # print('src:', src.shape)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].contiguous().view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# PREPROCESS DATA

train_src_filepath = os.path.join(args.train_data_path, "train_" + args.top_terms_strategy + ".source")
train_trg_filepath = os.path.join(args.train_data_path, "train_" + args.top_terms_strategy + ".target")

valid_src_filepath = os.path.join(args.train_data_path, "valid_" + args.top_terms_strategy + ".source")
valid_trg_filepath = os.path.join(args.train_data_path, "valid_" + args.top_terms_strategy + ".target")

test_src_filepath = os.path.join(args.train_data_path, "test_" + args.top_terms_strategy + ".source")
test_trg_filepath = os.path.join(args.train_data_path, "test_" + args.top_terms_strategy + ".target")


if 'tfidf' in args.top_terms_strategy:
    min_freq = 5
else:
    min_freq = 15

combined_vocab = build_combined_vocab([train_src_filepath, train_trg_filepath], tokenizer, freq=min_freq)
print("combined vocab:", len(combined_vocab))

if args.pretrained_emb is not None:
    vocab_file = os.path.join(args.save_path,
                              args.model_name + "_" + args.top_terms_strategy + "_emb.vocab")
else:
    vocab_file = os.path.join(args.save_path, args.model_name + "_" + args.top_terms_strategy + ".vocab")
model_file = vocab_file[:-6] + ".pt"

torch.save(combined_vocab, vocab_file)
print("Saved vocab at", vocab_file)

PAD_IDX = combined_vocab['<pad>']
SOS_IDX = combined_vocab['<sos>']
EOS_IDX = combined_vocab['<eos>']


vocab_size = len(combined_vocab)
emb_dim = 300
enc_hidden_dim = 200
dec_hidden_dim = 200
attn_dim = 64
drop = 0.5
BATCH_SIZE = 128

# load pretrained word embeddings
embeddings = None
if args.pretrained_emb is not None:
    embeddings = load_pretrained_embeddings(args.pretrained_emb, combined_vocab, emb_dim)

# load datasets
train_data = data_process(train_src_filepath, train_trg_filepath)
valid_data = data_process(valid_src_filepath, valid_trg_filepath)
test_data = data_process(test_src_filepath, test_trg_filepath)

train_iterator = DataLoader(train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=generate_batch)

valid_iterator = DataLoader(valid_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=generate_batch)

test_iterator = DataLoader(test_data,
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           collate_fn=generate_batch)

#initialize model
enc = Encoder(vocab_size,
              emb_dim,
              enc_hidden_dim,
              dec_hidden_dim,
              drop,
              embeddings)

attn = Attention(enc_hidden_dim,
                 dec_hidden_dim,
                 attn_dim)

dec = Decoder(vocab_size,
              emb_dim,
              enc_hidden_dim,
              dec_hidden_dim,
              drop,
              attn,
              embeddings)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

N_EPOCHS = 20
CLIP = 1

best_valid_loss = float('inf')

print("model filename:", model_file)
for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch + 1:02} of {N_EPOCHS}')
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_file)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')