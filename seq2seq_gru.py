# seq2seq_gru.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        # src: [batch, seq_len]
        embedded = self.embedding(src)  # [batch, seq_len, emb_dim]
        outputs, hidden = self.gru(embedded)  # hidden: [1, batch, hid_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_token, hidden):
        # input_token: [batch] of token IDs
        input_token = input_token.unsqueeze(1)    # [batch, 1]
        embedded = self.embedding(input_token)    # [batch, 1, emb_dim]
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))  # [batch, vocab_size]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, tgt):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        returns: outputs [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt.shape
        vocab_size = self.dec.fc_out.out_features

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # encode
        hidden = self.enc(src)

        # first input to decoder is <sos> tokens
        input_token = tgt[:, 0]  # [batch]

        for t in range(1, tgt_len):
            pred, hidden = self.dec(input_token, hidden)
            outputs[:, t] = pred
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs
