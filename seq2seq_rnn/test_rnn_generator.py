import os
import numpy as np
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
from torchtext.data.utils import get_tokenizer

from seq2seq_rnn.Seq2Seq import Seq2Seq, Encoder, Decoder, Attention
from seq2seq_rnn.rnn_utils import load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 1234
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = get_tokenizer('basic_english')

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', default='seq2seq_fi_tfidf.pt', type=str, help="saved model")
argparser.add_argument('--vocab_path', default='seq2seq_fi_tfidf_vocab', type=str, help="saved torchtext vocab")
argparser.add_argument('--topics_file', default='', type=str, help="text file with topic words")
argparser.add_argument('--output_path', default='', type=str, help="output")
argparser.add_argument('--num_labels', default=5, type=int, help="no. of labels to generate")
argparser.add_argument('--pretrained_emb', default=None, type=str, help="path to pretrained embeddings")
args = argparser.parse_args()

print("\n\n"+"-"*10, "Test RNN Seq2Seq model for Label Generation", "-"*10)
print("model_path:", args.model_path)
print("vocab_path:", args.vocab_path)
print("topics_file:", args.topics_file)
print("output_path:", args.output_path)
print("num_labels:", args.num_labels)
print("pretrained_emb:", args.pretrained_emb)
print("-"*70 + "\n\n")


def generate_label(sentence, vocab, model, device, max_len=10):
    tokens = [token.lower() for token in sentence]
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(src_indexes)
    src_tensor = torch.cat([torch.tensor([SOS_IDX]), src_tensor, torch.tensor([EOS_IDX])], dim=0).unsqueeze(0).to(device)
    output_indexes = []
    with torch.no_grad():
        src_tensor = src_tensor.permute(1, 0)
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
        # first input to the decoder is the <sos> token
        decoder_input = torch.LongTensor([SOS_IDX]).to(device)
        decoder_hidden = encoder_hidden
        for t in range(max_len):
            print('t:', t)
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_val, top_index = decoder_output.topk(1)
            decoder_input = torch.LongTensor([top_index]).to(device)
            out_index = int(torch.LongTensor([top_index]).to(device))
            output_indexes.append(out_index)
    generated_label = [vocab.itos[i] for i in output_indexes]
    print('output_indexes:', output_indexes)
    print('generated_label:', generated_label)
    return generated_label


# generate more than one label using beam search decoding
def generate_multiple_labels(sentence, vocab, model, device, beam_size=5, max_len=5):
    tokens = [token.lower() for token in sentence]
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(src_indexes)
    src_tensor = torch.cat([torch.tensor([SOS_IDX]), src_tensor, torch.tensor([EOS_IDX])], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        src_tensor = src_tensor.permute(1, 0)
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
        sequences = [[[SOS_IDX], 0.0]]
        beam_search_done = False
        # don't count SOS in the final output
        max_len = max_len + 1
        while beam_search_done is not True:
            all_candidates = []
            for i in range(len(sequences)):
                seq, score = sequences[i]
                # don't expand candidates whose last token is EOS or has reached max_len
                if seq[-1] == EOS_IDX or len(seq) == max_len:
                    all_candidates.append(sequences[i])
                else:
                    decoder_input = torch.LongTensor([SOS_IDX]).to(device)
                    decoder_hidden = encoder_hidden
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    #log_prob, indices = torch.topk(output, beam_size)
                    log_prob, indices = torch.topk(decoder_output, beam_size)
                    log_prob = log_prob.squeeze(0)
                    indices = indices.squeeze(0)
                    #print("log_prob:", log_prob)
                    #print("indices:", indices)
                    for j in range(beam_size):
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
        sent_tokens = [vocab.itos[i] for i in seq[1:]]
        # exclude SOS and EOS from final generated labels
        generated_labels.append((sent_tokens, score))
    return generated_labels

# load vocab
print("Loading vocab from", args.vocab_path)
vocab = torch.load(args.vocab_path)
print("model vocab:", len(vocab))
PAD_IDX = vocab['<pad>']
SOS_IDX = vocab['<sos>']
EOS_IDX = vocab['<eos>']
UNK_IDX = vocab['<unk>']

vocab_size = len(vocab)
emb_dim = 300
enc_hidden_dim = 200
dec_hidden_dim = 200
attn_dim = 64
drop = 0.5


# load pretrained word embeddings
embeddings = None
if args.pretrained_emb is not None:
    embeddings = load_pretrained_embeddings(args.pretrained_emb, vocab, emb_dim)


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
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# load saved model
model.load_state_dict(torch.load(args.model_path))
model.eval()


# test model on top words of trained LDA topics
print("Generating labels for topics in", args.topics_file)
topics = open(args.topics_file, 'r', encoding='utf-8').readlines()
topics = [t.strip() for t in topics]
print("Topics:", len(topics))

out_filename = args.model_path.split("/")[-1][:-3] + "_output.txt"
out_path = os.path.join(args.output_path, out_filename)
out_file = open(out_path, 'w', encoding='utf-8')
predicted_labels = []
with torch.no_grad():
    for k in range(len(topics)):
        print("-"*50)
        print("Topic", k+1, ":", topics[k] + "\n")
        sentence = [token for token in topics[k].split(', ')]
        pred_labels = generate_multiple_labels(sentence, vocab, model, device, beam_size=5, max_len=1)
        pred_labels = [pred[0][0] for pred in pred_labels]
        print('pred_labels:', pred_labels)
        pred_label_str = ', '.join(label for label in pred_labels if label != '<eos>')
        out_file.write(pred_label_str + '\n')

out_file.close()
print("Done! Predicted labels saved to", out_path, "!")




