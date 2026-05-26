import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import load_tess_dataset, get_splits, get_device, save_results
from models.speech_pipeline.train import SpeechDataset, SpeechEmotionModel, evaluate


def test(data_root, checkpoint='Results/speech_best_model.pt', results_dir='Results',
         batch_size=32):
    device = get_device()
    df = load_tess_dataset(data_root)
    _, _, te = get_splits(df)
    tel = DataLoader(SpeechDataset(te), batch_size, shuffle=False, num_workers=2)

    model = SpeechEmotionModel().to(device)
    assert os.path.exists(checkpoint), f'Checkpoint not found: {checkpoint}'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f'Loaded weights from {checkpoint}')

    crit = nn.CrossEntropyLoss()
    _, acc, yp, yt = evaluate(model, tel, crit, device)
    print(f'Speech Test Accuracy: {acc:.4f}')
    save_results(results_dir, 'speech_test', yt, yp)
    return acc, yt, yp


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   required=True)
    p.add_argument('--checkpoint',  default='Results/speech_best_model.pt')
    p.add_argument('--results_dir', default='Results')
    p.add_argument('--batch_size',  type=int, default=32)
    args = p.parse_args()
    test(args.data_root, args.checkpoint, args.results_dir, args.batch_size)