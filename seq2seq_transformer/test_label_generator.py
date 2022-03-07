import os
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

from seq2seq_transformer.Seq2Seq import Encoder, Decoder, Seq2Seq

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', default='trained_models', type=str, help="saved model")
argparser.add_argument('--vocab_path', default='', type=str, help="saved torchtext vocab")
argparser.add_argument('--topics_file', default='topics.txt', type=str, help="text file with topic words")
argparser.add_argument('--output_path', default='', type=str, help="output filename")
argparser.add_argument('--num_labels', default=5, type=int, help="no. of labels to generate")
# argparser.add_argument('--topics_csv', default=None, type=str, help="csv with topic words and gs labels")
args = argparser.parse_args()

print("\n\n"+"-"*10, "Test Seq2Seq transformers model for Label Generation", "-"*10)
print("model_path:", args.model_path)
print("vocab_path:", args.vocab_path)
print("output_path:", args.output_path)
print("topics_file:", args.topics_file)
print("num_labels:", args.num_labels)
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

tokenizer = get_tokenizer('basic_english')


# generate more than one label using beam search decoding
def generate_multiple_labels(sentence, vocab, model, device, beam_size=5, max_len=5):
    tokens = [token.lower() for token in sentence]
    tokens = [SOS_IDX] + tokens + [EOS_IDX]
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    sequences = [[[SOS_IDX], 0.0]]
    beam_search_done = False
    # don't count SOS and EOS in the final output
    max_len = max_len + 2
    while beam_search_done is not True:
        #print("hypotheses:", len(sequences))
        #print(sequences)
        all_candidates = []
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # print("expanding hypothesis", i+1, ":")
            # print("seq:", seq)
            # print("score:", score)
            # don't expand candidates whose last token is EOS or has reached max_len
            if seq[-1] == EOS_IDX or len(seq) == max_len:
                all_candidates.append(sequences[i])
            else:
                trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(trg_tensor)
                with torch.no_grad():
                    output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                log_prob, indices = torch.topk(output, beam_size)
                log_prob = log_prob.squeeze(0)[-1]
                indices = indices.squeeze(0)[-1]
                # print("log_prob:", log_prob)
                # print("indices:", indices)
                for j in range(beam_size):
                    if indices[j].item() != UNK_IDX:
                        normalised_score = (score + log_prob[j].item()) / (len(seq) + 1)
                        candidate = [seq + [indices[j].item()], normalised_score]
                        all_candidates.append(candidate)
        # sort candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]
        # check if all sequences end with EOS or is of max_len
        res = [True if sequences[i][0][-1] == EOS_IDX or len(sequences[i][0]) == max_len else False for i in range(len(sequences))]
        beam_search_done = False not in res
    generated_labels = []
    for i in range(len(sequences)):
        seq, score = sequences[i]
        sent_tokens = [vocab.itos[i] for i in seq]
        # exclude SOS and EOS from final generated labels
        generated_labels.append((sent_tokens[1:-1], score))
    return generated_labels



# load vocab
print("Loading vocab from", args.vocab_path)
vocab = torch.load(args.vocab_path)
print("model vocab:", len(vocab))
PAD_IDX = vocab['<pad>']
SOS_IDX = vocab['<sos>']
EOS_IDX = vocab['<eos>']
UNK_IDX = vocab['<unk>']

VOCAB_SIZE = len(vocab)
HID_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
PF_DIM = 512
DROPOUT = 0.1

LEARNING_RATE = 0.0001
N_EPOCHS = 30
CLIP = 1

enc = Encoder(VOCAB_SIZE,
              HID_DIM,
              NUM_LAYERS,
              NUM_HEADS,
              PF_DIM,
              DROPOUT,
              device)

dec = Decoder(VOCAB_SIZE,
              HID_DIM,
              NUM_LAYERS,
              NUM_HEADS,
              PF_DIM,
              DROPOUT,
              device)


model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# load saved model
model.load_state_dict(torch.load(args.model_path))
model.eval()

# test model on top words of trained LDA topics
print("Generating labels for topics in", args.topics_file)
topics = open(args.topics_file, 'r', encoding='utf-8').readlines()
topics = [t.strip() for t in topics]
print("Topics:", len(topics))

predicted_labels = []

with torch.no_grad():
    for k in range(len(topics)):
        print("-"*50)
        print("Topic", k+1, ":", topics[k] + "\n")
        sentence = [token for token in tokenizer(topics[k])]
        #label, attention = generate_label(sentence, vocab, model, device)
        pred_labels = generate_multiple_labels(sentence, vocab, model, device, beam_size=args.num_labels, max_len=1)
        pred_labels = [" ".join(labels[0]).strip() for labels in pred_labels]
        pred_labels = ", ".join(pred_labels)
        predicted_labels.append(pred_labels)
        print("Predicted:", pred_labels)

output_file = args.model_path.split("/")[-1][:-3] + "_output_" + args.label_method + ".txt"
final_out_path = os.path.join(args.output_path, output_file)
with open(final_out_path, 'w', encoding='utf-8') as f:
    for k in range(len(predicted_labels)):
        f.write(predicted_labels[k]+"\n")

print("Done! Saved final labels to", final_out_path)