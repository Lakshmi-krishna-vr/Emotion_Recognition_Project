import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import get_splits, get_device, save_results
from models.text_pipeline.train import TextDataset, TextEmotionModel, evaluate, CACHE_CSV


def test(checkpoint='Results/text_best_model.pt', results_dir='Results',
         batch_size=32, cache_csv=None, bert_name='bert-base-uncased'):
    if cache_csv is None:
        cache_csv = CACHE_CSV
    device = get_device()
    tok    = BertTokenizer.from_pretrained(bert_name)

    assert os.path.exists(cache_csv), f'Cache CSV not found: {cache_csv}'
    df = pd.read_csv(cache_csv)
    _, _, te = get_splits(df)
    tel = DataLoader(TextDataset(te, tok), batch_size, shuffle=False, num_workers=2)

    model = TextEmotionModel(bert_name).to(device)
    assert os.path.exists(checkpoint), f'Checkpoint not found: {checkpoint}'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f'Loaded weights from {checkpoint}')

    crit = nn.CrossEntropyLoss()
    _, acc, yp, yt = evaluate(model, tel, crit, device)
    print(f'Text Test Accuracy: {acc:.4f}')
    save_results(results_dir, 'text_test', yt, yp)
    return acc, yt, yp


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  default='Results/text_best_model.pt')
    p.add_argument('--results_dir', default='Results')
    p.add_argument('--cache_csv',   default=None)
    p.add_argument('--batch_size',  type=int, default=32)
    args = p.parse_args()
    test(args.checkpoint, args.results_dir, args.batch_size, args.cache_csv)