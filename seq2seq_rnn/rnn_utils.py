import torch
import numpy as np


def load_pretrained_embeddings(embeddings_path, vocab, embedding_dim=300):
    print("Loading pretrained embeddings from", embeddings_path)
    UNK_IDX = vocab['<unk>']
    with open(embeddings_path) as f:
        embeddings = np.random.rand(len(vocab), embedding_dim)
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = vocab.stoi[word]
            if index != UNK_IDX:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        embeddings = torch.from_numpy(embeddings).float()
        print("embeddings shape:",  embeddings.shape)
        return embeddings
