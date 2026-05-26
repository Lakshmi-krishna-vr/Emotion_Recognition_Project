"""
analyze.py
----------
Step A: Pre-compute prosody descriptions for all TESS samples and save to CSV.
Step B: Print a final accuracy summary from the Results/ directory.

Usage:
    # Pre-compute (run before training text / fusion pipelines)
    python analyze.py precompute --data_root data/TESS

    # Summary (run after all three pipelines have been trained)
    python analyze.py summary --results_dir Results
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_tess_dataset, load_audio,
    extract_prosody_cues, prosody_cues_to_text,
)


def precompute(data_root: str, cache_csv: str) -> None:
    if os.path.exists(cache_csv):
        print(f'Cache already exists: {cache_csv}  — skipping.')
        return

    df = load_tess_dataset(data_root)
    descriptions = []
    print('Extracting prosody cues (runs once)...')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        waveform, sr = load_audio(row['file_path'])
        cues = extract_prosody_cues(waveform, sr)
        descriptions.append(prosody_cues_to_text(cues))
    df['prosody_text'] = descriptions
    os.makedirs(os.path.dirname(cache_csv) or '.', exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f'Saved {len(df)} descriptions → {cache_csv}')

    print('\nSample descriptions:')
    for i in range(3):
        print(f'  [{df.iloc[i]["emotion"]}] {df.iloc[i]["prosody_text"]}')


def summary(results_dir: str) -> None:
    print('\n' + '=' * 50)
    print('OVERALL ACCURACY SUMMARY')
    print('=' * 50)
    for name in ['speech', 'text', 'fusion']:
        csv = os.path.join(results_dir, f'{name}_accuracy_table.csv')
        if os.path.exists(csv):
            df2 = pd.read_csv(csv, index_col=0)
            acc = float(df2.loc['accuracy', 'f1-score'])
            print(f'{name.capitalize():10s}: {acc * 100:.2f}%')
        else:
            print(f'{name.capitalize():10s}: (not found — train first)')


def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Analysis Tools')
    sub = parser.add_subparsers(dest='command', required=True)

    pc = sub.add_parser('precompute', help='Pre-compute prosody descriptions')
    pc.add_argument('--data_root',  required=True,
                    help='Path to TESS dataset root (folder containing emotion folders)')
    pc.add_argument('--cache_csv',  default='prosody_descriptions.csv')

    sm = sub.add_parser('summary', help='Print accuracy summary')
    sm.add_argument('--results_dir', default='Results')

    args = parser.parse_args()
    if args.command == 'precompute':
        precompute(args.data_root, args.cache_csv)
    elif args.command == 'summary':
        summary(args.results_dir)


if __name__ == '__main__':
    main()