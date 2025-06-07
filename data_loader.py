# data_loader.py

import json
import torch
from torch.utils.data import Dataset

class IITBDataset(Dataset):
    """
    Loads parallel Hindi–English lines for Seq2Seq training.
    Expects data/iitb/train.hi and train.en to exist.
    """
    def __init__(self,
                 hi_path="data/iitb/train.hi",
                 en_path="data/iitb/train.en",
                 hi_vocab="vocab/hindi_vocab.json",
                 en_vocab="vocab/english_vocab.json",
                 max_len=100):
        # read raw lines
        with open(hi_path, encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f if line.strip()]
        with open(en_path, encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f if line.strip()]

        # load vocabs
        with open(hi_vocab, encoding="utf-8") as f:
            self.src_stoi = json.load(f)
        with open(en_vocab, encoding="utf-8") as f:
            self.tgt_stoi = json.load(f)

        self.max_len = max_len
        assert len(self.src_lines) == len(self.tgt_lines), \
            "Source and target files must have same number of lines."

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = self.src_lines[idx].split()[: self.max_len - 2]
        tgt_tokens = self.tgt_lines[idx].split()[: self.max_len - 2]

        # add <sos> and <eos>
        src_ids = [self.src_stoi["<sos>"]] + \
                  [self.src_stoi.get(tok, self.src_stoi["<unk>"]) for tok in src_tokens] + \
                  [self.src_stoi["<eos>"]]
        tgt_ids = [self.tgt_stoi["<sos>"]] + \
                  [self.tgt_stoi.get(tok, self.tgt_stoi["<unk>"]) for tok in tgt_tokens] + \
                  [self.tgt_stoi["<eos>"]]

        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)

def collate_fn(batch):
    """
    Pads a batch of (src_ids, tgt_ids) to the max length in the batch.
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=0
    )
    return src_padded, tgt_padded
