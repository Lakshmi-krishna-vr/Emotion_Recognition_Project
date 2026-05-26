"""
Text Pipeline — FAST version
Reads pre-computed prosody descriptions from CSV.
No audio I/O during training.
Pipeline: prosody_text (pre-cached) → BertTokenizer → BERT → emotion
"""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import get_splits, save_results, set_seed, get_device, NUM_CLASSES

CACHE_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'prosody_descriptions.csv')


class TextDataset(Dataset):
    """Reads prosody_text from the pre-computed cache CSV."""
    def __init__(self, df, tokenizer, max_len=64):
        self.df      = df.reset_index(drop=True)
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = f"Spoken word: {row['transcript']}. Voice cues: {row['prosody_text']}"
        enc  = self.tok(
            text, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long),
        }


class TextEmotionModel(nn.Module):
    def __init__(self, bert_name='bert-base-uncased',
                 num_classes=NUM_CLASSES, dropout=0.3, unfreeze_last_n=12):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        total = len(self.bert.encoder.layer)
        for i in range(total - unfreeze_last_n, total):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(768, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return self.classifier(out.pooler_output)

    def get_representation(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return out.pooler_output


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum = correct = total = 0
    for b in tqdm(loader, desc='train', leave=False):
        ids   = b['input_ids'].to(device)
        attn  = b['attention_mask'].to(device)
        ttype = b['token_type_ids'].to(device)
        y     = b['label'].to(device)
        opt.zero_grad()
        logits = model(ids, attn, ttype)
        loss   = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    loss_sum = correct = total = 0
    preds, labels = [], []
    for b in loader:
        ids   = b['input_ids'].to(device)
        attn  = b['attention_mask'].to(device)
        ttype = b['token_type_ids'].to(device)
        y     = b['label'].to(device)
        logits = model(ids, attn, ttype)
        loss   = crit(logits, y)
        loss_sum += loss.item() * y.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        total   += y.size(0)
        preds.extend(p.cpu().tolist())
        labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, preds, labels


def train(data_root=None, epochs=10, batch_size=32, lr=2e-5,
          results_dir='Results', bert_name='bert-base-uncased',
          cache_csv=None):
    if cache_csv is None:
        cache_csv = CACHE_CSV
    set_seed(42)
    device = get_device()
    tok    = BertTokenizer.from_pretrained(bert_name)

    assert os.path.exists(cache_csv), \
        f'Cache CSV not found: {cache_csv}\nRun precompute_prosody.py first!'
    df_cached = pd.read_csv(cache_csv)

    tr, va, te = get_splits(df_cached)
    trl = DataLoader(TextDataset(tr, tok), batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    vl  = DataLoader(TextDataset(va, tok), batch_size, shuffle=False, num_workers=2)
    tel = DataLoader(TextDataset(te, tok), batch_size, shuffle=False, num_workers=2)

    model = TextEmotionModel(bert_name).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best  = 0.0
    spath = os.path.join(results_dir, 'text_best_model.pt')

    print('\n=== TEXT TRAINING  (Prosody-Text → BERT) ===')
    for ep in range(1, epochs + 1):
        tl, ta = train_epoch(model, trl, opt, crit, device)
        vl_, va_, _, _ = evaluate(model, vl, crit, device)
        sched.step()
        hist['train_loss'].append(tl);  hist['val_loss'].append(vl_)
        hist['train_acc'].append(ta);   hist['val_acc'].append(va_)
        print(f'Ep {ep:03d}/{epochs} | TrLoss {tl:.4f} TrAcc {ta:.4f} '
              f'| VaLoss {vl_:.4f} VaAcc {va_:.4f}')
        if va_ > best:
            best = va_
            torch.save(model.state_dict(), spath)
            print(f'  Saved best ({best:.4f})')

    model.load_state_dict(torch.load(spath, map_location=device))
    _, acc, yp, yt = evaluate(model, tel, crit, device)
    print(f'Test Accuracy: {acc:.4f}')
    save_results(results_dir, 'text', yt, yp, hist)
    return model


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   default=None)
    p.add_argument('--cache_csv',   default=None)
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=2e-5)
    p.add_argument('--results_dir', default='Results')
    args = p.parse_args()
    train(args.data_root, args.epochs, args.batch_size,
          args.lr, args.results_dir, cache_csv=args.cache_csv)