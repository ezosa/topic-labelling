import torch
import torch.nn as nn
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

from seq2seq_transformer.Seq2Seq import Encoder, Decoder, Seq2Seq

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='seq2seq_fi', type=str, help="model name")
argparser.add_argument('--save_path', default='project_dir/elaine/topic-labelling/yle/', type=str, help="save directory")
argparser.add_argument('--top_terms_strategy', default='tfidf', type=str, help="tfidf, sent, or combined")
args = argparser.parse_args()

print("\n\n"+"-"*10, "Train Seq2Seq Transformers for Label Generation", "-"*10)
print("model_name:", args.model_name)
print("save_path:", args.save_path)
print("top_terms_strategy:", args.top_terms_strategy)
print("-"*70 + "\n\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

SEED = 1234
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_test_pairs(src_filepath, trg_filepath):
    raw_src_iter = iter(io.open(src_filepath, encoding="utf8"))
    raw_trg_iter = iter(io.open(trg_filepath, encoding="utf8"))
    data = []
    for (raw_src, raw_trg) in zip(raw_src_iter, raw_trg_iter):
      src_tokens = [token for token in tokenizer(raw_src)]
      trg_tokens = [token for token in tokenizer(raw_trg)]
      data.append((src_tokens, trg_tokens))
    return data


def generate_label(sentence, vocab, model, device, max_len=50):
    model.eval()
    # if isinstance(sentence, str):
    #     nlp = spacy.load('de_core_news_sm')
    #     tokens = [token.text.lower() for token in nlp(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]
    tokens = [token.lower() for token in sentence]
    tokens = [vocab['<sos>']] + tokens + [vocab['<eos>']]
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [vocab['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == vocab['<eos>']:
            break
    trg_tokens = [vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


def generate_batch(data_batch):
    src_batch = []
    trg_batch = []
    for (src_item, trg_item) in data_batch:
        src_batch.append(torch.cat([torch.tensor([SOS_IDX]), src_item, torch.tensor([EOS_IDX])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([SOS_IDX]), trg_item, torch.tensor([EOS_IDX])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
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
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
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


train_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/train_" + args.top_terms_strategy + ".source"
train_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/train_" + args.top_terms_strategy + ".target"

valid_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/valid_" + args.top_terms_strategy + ".source"
valid_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/valid_" + args.top_terms_strategy + ".target"

test_src_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/test_" + args.top_terms_strategy + ".source"
test_trg_filepath = "/users/zosaelai/project_dir/elaine/topic-labelling/yle/test_" + args.top_terms_strategy + ".target"

tokenizer = get_tokenizer('spacy', language='en')
if args.top_terms_strategy == 'tfidf':
    min_freq = 5
elif args.top_terms_strategy == 'sent':
    min_freq = 15
else:
    min_freq = 25
combined_vocab = build_combined_vocab([train_src_filepath, train_trg_filepath], tokenizer, freq=min_freq)
print("combined vocab:", len(combined_vocab))

# save vocab
vocab_file = os.path.join(args.save_path, args.model_name + "_" + args.top_terms_strategy + "_vocab")
torch.save(combined_vocab, vocab_file)
print("Saved vocab at", vocab_file)

PAD_IDX = combined_vocab['<pad>']
SOS_IDX = combined_vocab['<sos>']
EOS_IDX = combined_vocab['<eos>']

print("PAD_IDX:", PAD_IDX)
print("SOS_IDX:", SOS_IDX)
print("EOS_IDX:", EOS_IDX)
print("-"*10)


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

INPUT_DIM = len(combined_vocab)
OUTPUT_DIM = len(combined_vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

LEARNING_RATE = 0.0005
N_EPOCHS = 20
CLIP = 1

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


model_file = os.path.join(args.save_path, args.model_name + "_" + args.top_terms_strategy + ".pt")
print("model file:", model_file)

model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
model.apply(initialize_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

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
        torch.save(model.state_dict(), model_file)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


print("Done training! Saved model at", model_file)

model.load_state_dict(torch.load(model_file))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


num_examples = 30
raw_test_data = get_test_pairs(test_src_filepath, test_trg_filepath)

example_indexes = [random.choice(range(len(raw_test_data))) for _ in range(num_examples)]

with torch.no_grad():
    for index in example_indexes:
        src = raw_test_data[index][0]
        trg = raw_test_data[index][1]
        print("-"*50)
        print(f'src = {" ".join(src)}')
        print(f'trg = {" ".join(trg)}')
        translation, attention = generate_label(src, combined_vocab, model, device)
        print(f'predicted trg = {" ".join(translation)}')



