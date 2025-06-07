# train_gru.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import IITBDataset, collate_fn
from seq2seq_gru import Encoder, Decoder, Seq2Seq

# Hyperparameters
EMB_DIM = 256
HID_DIM = 512
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10

# Paths
SRC_VOCAB = "vocab/hindi_vocab.json"
TGT_VOCAB = "vocab/english_vocab.json"
DATA_SRC = "data/iitb/train.hi"
DATA_TGT = "data/iitb/train.en"
CHECKPOINT_DIR = "/content/drive/MyDrive/AudioSummarizer/output/models/translator_gru"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Dataset & Loader
    ds = IITBDataset(DATA_SRC, DATA_TGT, SRC_VOCAB, TGT_VOCAB)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Model
    encoder = Encoder(len(ds.src_stoi), EMB_DIM, HID_DIM).to(DEVICE)
    decoder = Decoder(len(ds.tgt_stoi), EMB_DIM, HID_DIM).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for src_batch, tgt_batch in loader:
            src_batch, tgt_batch = src_batch.to(DEVICE), tgt_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch)
            # reshape for loss: skip first token
            output_dim = outputs.shape[-1]
            loss = criterion(
                outputs[:, 1:].reshape(-1, output_dim),
                tgt_batch[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:02d} — Loss: {avg_loss:.4f}")

        # save intermediate checkpoint
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("Training complete. Model saved to:", CHECKPOINT_PATH)

if __name__ == "__main__":
    train()
