import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
import numpy as np
import random
from seq2seq_transformer.Seq2Seq import Encoder, Decoder, Seq2Seq

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', default='/users/zosaelai/project_dir/elaine/topic-labelling/yle/seq2seq_fi_tfidf.pt', type=str, help="saved model")
argparser.add_argument('--vocab_path', default='/users/zosaelai/project_dir/elaine/topic-labelling/yle/seq2seq_fi_tfidf_vocab', type=str, help="saved torchtext vocab")
argparser.add_argument('--topics_file', default='/users/zosaelai/project_dir/elaine/topic-labelling/yle/yle_2018_lda_50topics_top_words.txt', type=str, help="text file with topic words")
args = argparser.parse_args()

print("\n\n"+"-"*10, "Test Seq2Seq transformers model for Label Generation", "-"*10)
print("model_path:", args.model_path)
print("vocab_path:", args.vocab_path)
print("topics_file:", args.topics_file)
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


# def beam_search_decoder(data, k, sentence, vocab, model, device, beam_size=2, max_len=5):
#     tokens = [token.lower() for token in sentence]
#     tokens = [vocab['<sos>']] + tokens + [vocab['<eos>']]
#     src_indexes = [vocab.stoi[token] for token in tokens]
#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
#     src_mask = model.make_src_mask(src_tensor)
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)
#     trg_indexes = [vocab['<sos>']] * beam_size
#     trg_scores = [0.0] * 5
#     sequences = [[list(), 0.0]]
#     # walk over each step in sequence
#     for row in data:
#         all_candidates = list()
#         for i in range(len(sequences)):
#             seq, score = sequences[i]
#             for j in range(len(row)):
#                 candidate = [seq + [j], score - torch.log(row[j])]
#                 all_candidates.append(candidate)
#         # sort candidates by score
#         ordered = sorted(all_candidates, key=lambda tup:tup[1])
#         sequences = ordered[:k]
#     return sequences


# generate more than one label using beam search decoding
def generate_multiple_labels(sentence, vocab, model, device, beam_size=3, max_len=5):
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
                    candidate = [seq + [indices[j].item()], score + log_prob[j].item()]
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
        generated_labels.append(sent_tokens[1:-1])
    return generated_labels


def generate_label(sentence, vocab, model, device, max_len=5):
    model.eval()
    tokens = [token.lower() for token in sentence]
    tokens = [SOS_IDX] + tokens + [EOS_IDX]
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [SOS_IDX]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        print("trg_tensor:", trg_tensor)
        print("output:", output)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == EOS_IDX:
            break
    trg_tokens = [vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


# load vocab
print("Loading vocab from", args.vocab_path)
vocab = torch.load(args.vocab_path)
print("model vocab:", len(vocab))
PAD_IDX = vocab['<pad>']
SOS_IDX = vocab['<sos>']
EOS_IDX = vocab['<eos>']


INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)
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


model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# load saved model
model.load_state_dict(torch.load(args.model_path))
model.eval()

# test model on top words of trained LDA topics
topics = open(args.topics_file, 'r', encoding='utf-8').readlines()
topics = [t.strip() for t in topics]
print("Topics:", len(topics))
tokenizer = get_tokenizer('spacy', language='en')
with torch.no_grad():
    for k in range(len(topics)):
        print("-"*50)
        print("Topic", k+1, ":", topics[k])
        sentence = [token for token in tokenizer(topics[k])]
        #label, attention = generate_label(sentence, vocab, model, device)
        labels = generate_multiple_labels(sentence, vocab, model, device, beam_size=3, max_len=5)
        for i, label in enumerate(labels):
            print(f'\nPredicted Label {i+1} : {" ".join(label)}')



