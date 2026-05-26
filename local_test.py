"""
local_test.py
-------------
Quick smoke-test: load a single WAV file and run it through each trained pipeline.

Usage:
    python local_test.py --audio path/to/file.wav \
                         --speech_ckpt Results/speech_best_model.pt \
                         --text_ckpt   Results/text_best_model.pt \
                         --fusion_ckpt Results/fusion_best_model.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_audio, extract_mfcc, extract_prosody_cues,
    prosody_cues_to_text, IDX_TO_EMOTION, get_device,
)
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train   import TextEmotionModel
from models.fusion_pipeline.train import FusionEmotionModel


MAX_LEN_AUDIO = 345
MAX_LEN_TEXT  = 64


def predict_speech(model, waveform, sr, device):
    feat = extract_mfcc(waveform, sr=sr)
    T = feat.shape[0]
    if T < MAX_LEN_AUDIO:
        feat = np.vstack([feat, np.zeros((MAX_LEN_AUDIO - T, feat.shape[1]))])
    else:
        feat = feat[:MAX_LEN_AUDIO]
    mean = feat.mean(0, keepdims=True)
    std  = feat.std(0,  keepdims=True) + 1e-8
    x = torch.tensor((feat - mean) / std, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze()
    pred  = logits.argmax(1).item()
    return IDX_TO_EMOTION[pred], probs.cpu().numpy()


def predict_text(model, tokenizer, combined_text, device):
    enc = tokenizer(
        combined_text, max_length=MAX_LEN_TEXT,
        padding='max_length', truncation=True, return_tensors='pt')
    ids   = enc['input_ids'].to(device)
    attn  = enc['attention_mask'].to(device)
    ttype = enc['token_type_ids'].to(device)
    with torch.no_grad():
        logits = model(ids, attn, ttype)
    probs = torch.softmax(logits, dim=1).squeeze()
    pred  = logits.argmax(1).item()
    return IDX_TO_EMOTION[pred], probs.cpu().numpy()


def predict_fusion(model, tokenizer, waveform, sr, combined_text, device):
    feat = extract_mfcc(waveform, sr=sr)
    T = feat.shape[0]
    if T < MAX_LEN_AUDIO:
        feat = np.vstack([feat, np.zeros((MAX_LEN_AUDIO - T, feat.shape[1]))])
    else:
        feat = feat[:MAX_LEN_AUDIO]
    mean = feat.mean(0, keepdims=True)
    std  = feat.std(0,  keepdims=True) + 1e-8
    sp = torch.tensor((feat - mean) / std, dtype=torch.float32).unsqueeze(0).to(device)

    enc = tokenizer(
        combined_text, max_length=MAX_LEN_TEXT,
        padding='max_length', truncation=True, return_tensors='pt')
    ids   = enc['input_ids'].to(device)
    attn  = enc['attention_mask'].to(device)
    ttype = enc['token_type_ids'].to(device)

    with torch.no_grad():
        logits = model(sp, ids, attn, ttype)
    probs = torch.softmax(logits, dim=1).squeeze()
    pred  = logits.argmax(1).item()
    return IDX_TO_EMOTION[pred], probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Run a single WAV through all pipelines')
    parser.add_argument('--audio',        required=True, help='Path to WAV file')
    parser.add_argument('--speech_ckpt',  default='Results/speech_best_model.pt')
    parser.add_argument('--text_ckpt',    default='Results/text_best_model.pt')
    parser.add_argument('--fusion_ckpt',  default='Results/fusion_best_model.pt')
    args = parser.parse_args()

    assert os.path.exists(args.audio), f'Audio file not found: {args.audio}'
    device = get_device()

    print(f'\nLoading audio: {args.audio}')
    waveform, sr = load_audio(args.audio)
    
    # ── Parse Transcript Word from Filename ──
    # TESS files look like 'OAF_back_fear.wav' -> word is parts[1:-1]
    name_stem = Path(args.audio).stem.lower()
    parts = name_stem.split('_')
    if len(parts) >= 3:
        transcript_word = '_'.join(parts[1:-1])
    else:
        transcript_word = "unknown"  # Fallback if manual filepath format varies
    
    # ── Process Prosody Description ──
    cues         = extract_prosody_cues(waveform, sr)
    prosody_text = prosody_cues_to_text(cues)
    
    # ── UPDATED: Match combined text structure used in training ──
    combined_input_text = f"Spoken word: {transcript_word}. Voice cues: {prosody_text}"
    print(f"Parsed Word       : {transcript_word}")
    print(f"Combined Text Input: {combined_input_text}\n")

    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    emotions = list(IDX_TO_EMOTION.values())
    pad = max(len(e) for e in emotions)

    # ── Speech ──
    if os.path.exists(args.speech_ckpt):
        model = SpeechEmotionModel().to(device)
        model.load_state_dict(torch.load(args.speech_ckpt, map_location=device))
        model.eval()
        pred, probs = predict_speech(model, waveform, sr, device)
        print(f'[SPEECH]  Prediction: {pred}')
        for e, p in zip(emotions, probs):
            print(f'  {e:<{pad}}: {p * 100:5.1f}%')
    else:
        print(f'[SPEECH]  checkpoint not found: {args.speech_ckpt}')

    print()

    # ── Text ──
    if os.path.exists(args.text_ckpt):
        model = TextEmotionModel(unfreeze_last_n=12).to(device)
        model.load_state_dict(torch.load(args.text_ckpt, map_location=device))
        model.eval()
        pred, probs = predict_text(model, tok, combined_input_text, device)
        print(f'[TEXT]    Prediction: {pred}')
        for e, p in zip(emotions, probs):
            print(f'  {e:<{pad}}: {p * 100:5.1f}%')
    else:
        print(f'[TEXT]    checkpoint not found: {args.text_ckpt}')

    print()

    # ── Fusion ──
    if os.path.exists(args.fusion_ckpt):
        model = FusionEmotionModel().to(device)
        model.load_state_dict(torch.load(args.fusion_ckpt, map_location=device))
        model.eval()
        pred, probs = predict_fusion(model, tok, waveform, sr, combined_input_text, device)
        print(f'[FUSION]  Prediction: {pred}')
        for e, p in zip(emotions, probs):
            print(f'  {e:<{pad}}: {p * 100:5.1f}%')
    else:
        print(f'[FUSION]  checkpoint not found: {args.fusion_ckpt}')


if __name__ == '__main__':
    main()