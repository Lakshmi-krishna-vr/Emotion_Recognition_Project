"""
Fusion Pipeline — FAST version
Speech branch : audio → MFCC → BiLSTM
Text branch   : reads pre-cached prosody_text → BERT (no audio I/O)
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import (get_splits, load_audio, extract_mfcc,
                   save_results, set_seed, get_device, NUM_CLASSES)
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train   import TextEmotionModel

CACHE_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'prosody_descriptions.csv')


class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer,
                 sr=22050, duration=4.0,
                 max_len_audio=345, max_len_text=64):
        self.df  = df.reset_index(drop=True)
        self.tok = tokenizer
        self.sr  = sr
        self.duration = duration
        self.mla = max_len_audio
        self.mlt = max_len_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Speech branch ──
        waveform, sr = load_audio(row['file_path'], sr=self.sr, duration=self.duration)
        feat = extract_mfcc(waveform, sr=sr)
        T = feat.shape[0]
        if T < self.mla:
            feat = np.vstack([feat,
                np.zeros((self.mla - T, feat.shape[1]), dtype=np.float32)])
        else:
            feat = feat[:self.mla]
        mean = feat.mean(0, keepdims=True)
        std  = feat.std(0,  keepdims=True) + 1e-8
        feat = (feat - mean) / std

        # ── Text branch ──
        text = str(row['prosody_text'])
        enc  = self.tok(
            text, max_length=self.mlt,
            padding='max_length', truncation=True, return_tensors='pt')
        return {
            'speech':         torch.tensor(feat, dtype=torch.float32),
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'label':          torch.tensor(row['label'], dtype=torch.long),
        }


class GatedFusion(nn.Module):
    def __init__(self, sd=512, td=768, fd=512):
        super().__init__()
        self.ps   = nn.Linear(sd, fd)
        self.pt   = nn.Linear(td, fd)
        self.gate = nn.Linear(sd + td, fd)

    def forward(self, s, t):
        sp = torch.tanh(self.ps(s))
        tp = torch.tanh(self.pt(t))
        g  = torch.sigmoid(self.gate(torch.cat([s, t], dim=1)))
        return g * sp + (1 - g) * tp


class FusionEmotionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.speech_enc = SpeechEmotionModel()
        self.text_enc   = TextEmotionModel()
        self.speech_enc.classifier = nn.Identity()
        self.text_enc.classifier   = nn.Identity()
        self.fusion = GatedFusion()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, speech, input_ids, attention_mask, token_type_ids=None):
        s = self.speech_enc.get_representation(speech)
        t = self.text_enc.get_representation(input_ids, attention_mask, token_type_ids)
        return self.classifier(self.fusion(s, t))

    def get_representation(self, speech, input_ids, attention_mask, token_type_ids=None):
        s = self.speech_enc.get_representation(speech)
        t = self.text_enc.get_representation(input_ids, attention_mask, token_type_ids)
        return self.fusion(s, t)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum = correct = total = 0
    for b in tqdm(loader, desc='train', leave=False):
        sp    = b['speech'].to(device)
        ids   = b['input_ids'].to(device)
        attn  = b['attention_mask'].to(device)
        ttype = b['token_type_ids'].to(device)
        y     = b['label'].to(device)
        opt.zero_grad()
        logits = model(sp, ids, attn, ttype)
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
        sp    = b['speech'].to(device)
        ids   = b['input_ids'].to(device)
        attn  = b['attention_mask'].to(device)
        ttype = b['token_type_ids'].to(device)
        y     = b['label'].to(device)
        logits = model(sp, ids, attn, ttype)
        loss   = crit(logits, y)
        loss_sum += loss.item() * y.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        total   += y.size(0)
        preds.extend(p.cpu().tolist())
        labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, preds, labels


def train(data_root=None,
          speech_ckpt='Results/speech_best_model.pt',
          text_ckpt='Results/text_best_model.pt',
          epochs=20, batch_size=16, lr=5e-5,
          results_dir='Results', cache_csv=None):
    if cache_csv is None:
        cache_csv = CACHE_CSV
    set_seed(42)
    device = get_device()
    tok    = BertTokenizer.from_pretrained('bert-base-uncased')

    assert os.path.exists(cache_csv), \
        f'Cache not found: {cache_csv}\nRun precompute_prosody.py first!'
    df = pd.read_csv(cache_csv)

    tr, va, te = get_splits(df)
    trl = DataLoader(MultimodalDataset(tr, tok), batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    vl  = DataLoader(MultimodalDataset(va, tok), batch_size, shuffle=False, num_workers=2)
    tel = DataLoader(MultimodalDataset(te, tok), batch_size, shuffle=False, num_workers=2)

    model = FusionEmotionModel().to(device)
    if os.path.exists(speech_ckpt):
        st = {k: v for k, v in torch.load(speech_ckpt, map_location=device).items()
              if not k.startswith('classifier')}
        model.speech_enc.load_state_dict(st, strict=False)
        print(f'Loaded speech encoder from {speech_ckpt}')
    if os.path.exists(text_ckpt):
        st = {k: v for k, v in torch.load(text_ckpt, map_location=device).items()
              if not k.startswith('classifier')}
        model.text_enc.load_state_dict(st, strict=False)
        print(f'Loaded BERT text encoder from {text_ckpt}')

    crit   = nn.CrossEntropyLoss()
    bert_p = list(model.text_enc.bert.parameters())
    bert_ids = {id(bp) for bp in bert_p}
    other_p  = [p for p in model.parameters() if id(p) not in bert_ids]
    opt = torch.optim.AdamW(
        [{'params': bert_p,  'lr': lr * 0.1},
         {'params': other_p, 'lr': lr}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best  = 0.0
    spath = os.path.join(results_dir, 'fusion_best_model.pt')

    print('\n=== FUSION TRAINING  (BiLSTM + Prosody-Text BERT) ===')
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
    save_results(results_dir, 'fusion', yt, yp, hist)
    return model


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',    default=None)
    p.add_argument('--cache_csv',    default=None)
    p.add_argument('--speech_ckpt',  default='Results/speech_best_model.pt')
    p.add_argument('--text_ckpt',    default='Results/text_best_model.pt')
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=5e-5)
    p.add_argument('--results_dir',  default='Results')
    args = p.parse_args()
    train(args.data_root, args.speech_ckpt, args.text_ckpt,
          args.epochs, args.batch_size, args.lr,
          args.results_dir, args.cache_csv)