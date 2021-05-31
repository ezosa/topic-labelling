import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

import spacy
import numpy as np

import random
import math
import time

import io
from collections import Counter

from seq2seq_transformer.Seq2Seq import Encoder, Decoder, Seq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

SEED = 1234
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# spacy_de = spacy.load('de_core_news_sm')
# spacy_en = spacy.load('en_core_web_sm')

def get_test_pairs(src_filepath, trg_filepath):
    raw_src_iter = iter(io.open(src_filepath, encoding="utf8"))
    raw_trg_iter = iter(io.open(trg_filepath, encoding="utf8"))
    data = []
    for (raw_src, raw_trg) in zip(raw_src_iter, raw_trg_iter):
      src_tokens = [token for token in tokenizer(raw_src)]
      trg_tokens = [token for token in tokenizer(raw_trg)]
      data.append((src_tokens, trg_tokens))
    return data


def generate_label(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    # if isinstance(sentence, str):
    #     nlp = spacy.load('de_core_news_sm')
    #     tokens = [token.text.lower() for token in nlp(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]
    tokens = [token.lower() for token in sentence]

    tokens = [src_vocab['<sos>']] + tokens + [src_vocab['<eos>']]
    src_indexes = [src_vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def generate_batch(data_batch):
    src_batch = []
    trg_batch = []
    #print("data_batch:", len(data_batch))
    for (src_item, trg_item) in data_batch:
        #print("src_item:", src_item)
        #print("trg_item:", trg_item)
        src_batch.append(torch.cat([torch.tensor([SRC_SOS_IDX]), src_item, torch.tensor([SRC_EOS_IDX])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([TRG_SOS_IDX]), trg_item, torch.tensor([TRG_EOS_IDX])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=TRG_PAD_IDX, batch_first=True)
    # print("src_batch:", src_batch.shape)
    # print("trg_batch:", trg_batch.shape)
    return src_batch, trg_batch


def build_vocab(filepath, tokenizer):
    counter = Counter()
    lines = open(filepath, 'r', encoding="utf8").readlines()
    for line in lines:
        counter.update(tokenizer(line.lower()))
    return Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=2)


def data_process(src_filepath, trg_filepath):
    raw_src_iter = iter(io.open(src_filepath, encoding="utf8"))
    raw_trg_iter = iter(io.open(trg_filepath, encoding="utf8"))
    data = []
    for (raw_src, raw_trg) in zip(raw_src_iter, raw_trg_iter):
      src_tensor = torch.tensor([src_vocab[token] for token in tokenizer(raw_src)], dtype=torch.long)
      trg_tensor = torch.tensor([trg_vocab[token] for token in tokenizer(raw_trg)], dtype=torch.long)
      data.append((src_tensor, trg_tensor))
    return data


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# def tokenize_de(text):
#     return [tok.text for tok in spacy_de.tokenizer(text)]
#
#
# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        # print("src:", src.shape)
        # print(src)
        # print("trg:", trg.shape)
        # print(trg)
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
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
            src = src.to(device)
            trg = trg.to(device)

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# SRC = Field(tokenize=tokenize_de,
#             init_token='<sos>',
#             eos_token='<eos>',
#             lower=True,
#             batch_first=True)
#
# TRG = Field(tokenize=tokenize_en,
#             init_token='<sos>',
#             eos_token='<eos>',
#             lower=True,
#             batch_first=True)
#
#
# train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
#                                                     fields=(SRC, TRG))
#
# SRC.build_vocab(train_data, min_freq=2)
# TRG.build_vocab(train_data, min_freq=2)

train_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/train_tfidf.source"
train_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/train_tfidf.target"

valid_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/valid_tfidf.source"
valid_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/valid_tfidf.target"

test_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/test_tfidf.source"
test_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/wiki/test_tfidf.target"

tokenizer = get_tokenizer('spacy', language='en')
src_vocab = build_vocab(train_src_filepath, tokenizer)
trg_vocab = build_vocab(train_trg_filepath, tokenizer)

print("src vocab:", len(src_vocab))
print("trg vocab:", len(trg_vocab))

SRC_PAD_IDX = src_vocab['<pad>']
SRC_SOS_IDX = src_vocab['<sos>']
SRC_EOS_IDX = src_vocab['<eos>']

TRG_PAD_IDX = trg_vocab['<pad>']
TRG_SOS_IDX = trg_vocab['<sos>']
TRG_EOS_IDX = trg_vocab['<eos>']

print("SRC PAD_IDX:", SRC_PAD_IDX)
print("SRC SOS_IDX:", SRC_SOS_IDX)
print("SRC EOS_IDX:", SRC_EOS_IDX)
print("-"*10)
print("TRG PAD_IDX:", TRG_PAD_IDX)
print("TRG SOS_IDX:", TRG_SOS_IDX)
print("TRG EOS_IDX:", TRG_EOS_IDX)


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

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#      batch_size=BATCH_SIZE,
#      device=device)


INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)


# SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# print("SRC_PAD_IDX:", SRC_PAD_IDX)
# print("TRG_PAD_IDX:", TRG_PAD_IDX)
#
# print("SRC init_token index:", SRC.vocab.stoi[SRC.init_token])
# print("SRC eos_token index:", SRC.vocab.stoi[SRC.eos_token])
#
# print("TRG init_token index:", TRG.vocab.stoi[TRG.init_token])
# print("TRG eos_token index:", TRG.vocab.stoi[TRG.eos_token])

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)


LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

N_EPOCHS = 20
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch + 1:02} of {N_EPOCHS}')
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'seq2seq-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('seq2seq-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


num_examples = 20
raw_test_data = get_test_pairs(test_src_filepath, test_trg_filepath)

example_indexes = [random.choice(range(len(raw_test_data))) for _ in range(num_examples)]

with torch.no_grad():
    for index in example_indexes:
        src = raw_test_data[index][0]
        trg = raw_test_data[index][1]
        print("-"*50)
        print(f'src = {" ".join(src)}')
        print(f'trg = {" ".join(trg)}')
        translation, attention = generate_label(src, src_vocab, trg_vocab, model, device)
        print(f'predicted trg = {" ".join(translation)}')



