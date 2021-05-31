from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DataLoader(Dataset):
    def __init__(self, source_file, target_file, word2id):
        sources = open(source_file, 'r', encoding='utf-8').readlines()
        self.sources = [s.strip() for s in sources]
        targets = open(target_file, 'r', encoding='utf-8').readlines()
        self.targets = [t.strip() for t in targets]
        self.word2id = word2id

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        source = self.sources[idx].split()
        target = self.targets[idx].split()
        source = np.array([])
        row = dict(self.data.iloc()[idx])
        row['id'] = str(row['id'])
        combined_text = row['headline'] + ' ' + row['text']
        combined_text = combined_text.lower().split()
        combined_text = [self.vocab[w] if w in self.vocab else self.vocab['OOV'] for w in combined_text]
        if len(combined_text) > self.max_text_len:
                combined_text = combined_text[:self.max_text_len]
        else:
            combined_text.extend([self.vocab['OOV']]*(self.max_text_len-len(combined_text)))
        row['content'] = combined_text
        row['content_len'] = self.max_text_len
        if 'codes' in row:
            codes = row['codes']
            codes = codes.split()
            codes2id = np.array([self.label2id[code] for code in codes if code in self.label2id])
            binary_label = np.zeros(self.labels_len)
            binary_label[codes2id] = 1
            row['binary_label'] = binary_label
        return row


