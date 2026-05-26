"""
test_models.py
==============
Comprehensive test script to evaluate emotion recognition models.
Tests speech, text, and fusion models and saves accuracy metrics + plots.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import (
    EMOTION_LABELS, NUM_CLASSES, extract_mfcc, load_audio,
    set_seed, get_device, IDX_TO_EMOTION, load_tess_dataset, get_splits
)

LABEL_NAMES = EMOTION_LABELS
TX_MAX_LEN = 64
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train import TextEmotionModel
from models.fusion_pipeline.train import FusionEmotionModel


def extract_speech_features(audio_path: str, sr: int = 22050, duration: float = 4.0):
    """Extract MFCC features from audio file."""
    waveform, _ = load_audio(audio_path, sr=sr, duration=duration)
    return extract_mfcc(waveform, sr=sr, n_mfcc=40, n_fft=512, hop_length=256)


def load_models(device):
    """Load all three models (speech, text, fusion)."""
    results_dir = ROOT / 'Results'
    
    speech_weights = results_dir / 'speech_best_model.pt'
    text_weights = results_dir / 'text_best_model.pt'
    fusion_weights = results_dir / 'fusion_best_model.pt'
    
    print("Loading models...")
    
    # Speech model
    speech_model = SpeechEmotionModel()
    if speech_weights.exists():
        speech_model.load_state_dict(torch.load(speech_weights, map_location=device))
        print(f"✓ Loaded speech model from {speech_weights}")
    else:
        print(f"✗ Speech weights not found at {speech_weights}")
    speech_model = speech_model.to(device).eval()
    
    # Text model
    text_model = TextEmotionModel()
    if text_weights.exists():
        text_model.load_state_dict(torch.load(text_weights, map_location=device))
        print(f"✓ Loaded text model from {text_weights}")
    else:
        print(f"✗ Text weights not found at {text_weights}")
    text_model = text_model.to(device).eval()
    
    # Fusion model (creates its own sub-models)
    fusion_model = FusionEmotionModel()
    if fusion_weights.exists():
        fusion_model.load_state_dict(torch.load(fusion_weights, map_location=device))
        print(f"✓ Loaded fusion model from {fusion_weights}")
    else:
        print(f"✗ Fusion weights not found at {fusion_weights}")
    fusion_model = fusion_model.to(device).eval()
    
    return speech_model, text_model, fusion_model


def evaluate_speech_model(model, test_df, device, sr=22050, duration=4.0, max_samples=None):
    """Evaluate speech model on test dataset."""
    print("\n" + "="*60)
    print("EVALUATING SPEECH MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    failed = 0
    
    test_data = test_df if max_samples is None else test_df.head(max_samples)
    total = len(test_data)
    
    for idx, row in test_data.iterrows():
        try:
            audio_feat = extract_speech_features(row['file_path'], sr=sr, duration=duration)
            
            # Ensure shape is (T, n_features)
            if audio_feat.ndim == 2:
                feat = audio_feat
            else:
                feat = audio_feat.reshape(-1, 1)
            
            # Convert to tensor and add batch dimension
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Pad/truncate to expected length (345 timesteps)
            expected_len = 345
            if feat_tensor.shape[1] < expected_len:
                pad_len = expected_len - feat_tensor.shape[1]
                feat_tensor = F.pad(feat_tensor, (0, 0, 0, pad_len))
            else:
                feat_tensor = feat_tensor[:, :expected_len, :]
            
            with torch.no_grad():
                logits = model(feat_tensor)
                pred = logits.argmax(dim=1).item()
            
            y_true.append(row['label'])
            y_pred.append(pred)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} samples")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ✗ Failed on {row['file_path']}: {str(e)[:60]}")
    
    if failed > 0 and failed > 3:
        print(f"  ... {failed - 3} more failures")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Speech Model - Accuracy: {accuracy:.4f} ({len(y_true)}/{total} processed)")
    
    return np.array(y_true), np.array(y_pred), accuracy


def evaluate_text_model(model, test_df, device, tokenizer, max_samples=None):
    """Evaluate text model on test dataset."""
    print("\n" + "="*60)
    print("EVALUATING TEXT MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    failed = 0
    
    test_data = test_df if max_samples is None else test_df.head(max_samples)
    total = len(test_data)
    
    for idx, row in test_data.iterrows():
        try:
            # Use transcript word as text input
            text_input = row['transcript']
            if pd.isna(text_input):
                text_input = "unknown"
            
            enc = tokenizer(
                str(text_input),
                padding='max_length',
                truncation=True,
                max_length=TX_MAX_LEN,
                return_tensors='pt'
            )
            
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            token_type_ids = enc.get('token_type_ids', torch.zeros_like(ids)).to(device)
            
            with torch.no_grad():
                logits = model(ids, mask, token_type_ids)
                pred = logits.argmax(dim=1).item()
            
            y_true.append(row['label'])
            y_pred.append(pred)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} samples")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ✗ Failed on sample {idx}: {str(e)[:60]}")
    
    if failed > 0 and failed > 3:
        print(f"  ... {failed - 3} more failures")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Text Model - Accuracy: {accuracy:.4f} ({len(y_true)}/{total} processed)")
    
    return np.array(y_true), np.array(y_pred), accuracy


def evaluate_fusion_model(model, test_df, device, tokenizer, sr=22050, duration=4.0, max_samples=None):
    """Evaluate fusion model on test dataset."""
    print("\n" + "="*60)
    print("EVALUATING FUSION MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    failed = 0
    
    test_data = test_df if max_samples is None else test_df.head(max_samples)
    total = len(test_data)
    
    for idx, row in test_data.iterrows():
        try:
            # Speech features
            audio_feat = extract_speech_features(row['file_path'], sr=sr, duration=duration)
            if audio_feat.ndim == 2:
                feat = audio_feat
            else:
                feat = audio_feat.reshape(-1, 1)
            
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            expected_len = 345
            if feat_tensor.shape[1] < expected_len:
                pad_len = expected_len - feat_tensor.shape[1]
                feat_tensor = F.pad(feat_tensor, (0, 0, 0, pad_len))
            else:
                feat_tensor = feat_tensor[:, :expected_len, :]
            
            # Text features
            text_input = row['transcript']
            if pd.isna(text_input):
                text_input = "unknown"
            
            enc = tokenizer(
                str(text_input),
                padding='max_length',
                truncation=True,
                max_length=TX_MAX_LEN,
                return_tensors='pt'
            )
            
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            token_type_ids = enc.get('token_type_ids', torch.zeros_like(ids)).to(device)
            
            with torch.no_grad():
                logits = model(feat_tensor, ids, mask, token_type_ids)
                pred = logits.argmax(dim=1).item()
            
            y_true.append(row['label'])
            y_pred.append(pred)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} samples")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ✗ Failed on {row['file_path']}: {str(e)[:60]}")
    
    if failed > 0 and failed > 3:
        print(f"  ... {failed - 3} more failures")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Fusion Model - Accuracy: {accuracy:.4f} ({len(y_true)}/{total} processed)")
    
    return np.array(y_true), np.array(y_pred), accuracy


def save_confusion_matrix_plot(y_true, y_pred, model_name, output_dir):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


def save_accuracy_per_emotion_plot(y_true, y_pred, model_name, output_dir):
    """Save per-emotion accuracy bar plot."""
    accuracies = []
    for emotion_idx in range(len(LABEL_NAMES)):
        mask = y_true == emotion_idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == emotion_idx).sum() / mask.sum()
            accuracies.append(acc)
        else:
            accuracies.append(0.0)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(LABEL_NAMES)))
    bars = plt.bar(LABEL_NAMES, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title(f'{model_name} - Accuracy per Emotion', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{model_name}_accuracy_per_emotion.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


def save_accuracy_table(y_true, y_pred, model_name, output_dir):
    """Save accuracy metrics as CSV table."""
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    output_file = os.path.join(output_dir, f'{model_name}_accuracy_table.csv')
    report_df.to_csv(output_file)
    print(f"  ✓ Saved: {output_file}")
    
    # Also print to console
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    
    return report_df


def save_comparison_plot(results, output_dir):
    """Save side-by-side model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (model_name, (y_true, y_pred)) in enumerate(results.items()):
        # Calculate per-emotion accuracies
        accuracies = []
        for emotion_idx in range(len(LABEL_NAMES)):
            mask = y_true == emotion_idx
            if mask.sum() > 0:
                acc = (y_pred[mask] == emotion_idx).sum() / mask.sum()
                accuracies.append(acc)
            else:
                accuracies.append(0.0)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(LABEL_NAMES)))
        axes[idx].bar(LABEL_NAMES, accuracies, color=colors, edgecolor='black', linewidth=1)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Accuracy', fontsize=10)
        axes[idx].set_ylim(0, 1.1)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Emotion Recognition Models - Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def save_accuracy_matrix(results, output_dir):
    """Save overall accuracy matrix as heatmap."""
    model_names = ['Speech', 'Text', 'Fusion']
    accuracies_by_emotion = {}
    
    for idx, model_name in enumerate(list(results.keys())):
        y_true, y_pred = results[model_name]
        accuracies = []
        for emotion_idx in range(len(LABEL_NAMES)):
            mask = y_true == emotion_idx
            if mask.sum() > 0:
                acc = (y_pred[mask] == emotion_idx).sum() / mask.sum()
                accuracies.append(acc * 100)  # Convert to percentage
            else:
                accuracies.append(0.0)
        accuracies_by_emotion[model_names[idx]] = accuracies
    
    df_accuracy = pd.DataFrame(accuracies_by_emotion, index=LABEL_NAMES)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df_accuracy, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=100,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    plt.title('Model Accuracy Matrix (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'accuracy_matrix.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")
    
    # Also save as CSV
    csv_file = os.path.join(output_dir, 'accuracy_matrix.csv')
    df_accuracy.to_csv(csv_file)
    print(f"✓ Saved: {csv_file}")
    
    return df_accuracy


def save_summary_report(results, output_dir):
    """Save summary report with overall accuracies."""
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EMOTION RECOGNITION MODELS - EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        model_names = ['Speech', 'Text', 'Fusion']
        for idx, (model_key, (y_true, y_pred)) in enumerate(results.items()):
            model_name = model_names[idx]
            accuracy = accuracy_score(y_true, y_pred)
            
            f.write(f"\n{model_name.upper()} MODEL\n")
            f.write("-" * 70 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy*len(y_true))}/{len(y_true)})\n")
            f.write(f"\nPer-Emotion Accuracy:\n")
            
            for emotion_idx, emotion in enumerate(LABEL_NAMES):
                mask = y_true == emotion_idx
                if mask.sum() > 0:
                    acc = (y_pred[mask] == emotion_idx).sum() / mask.sum()
                    f.write(f"  {emotion:12s}: {acc:6.2%} ({int(acc*mask.sum())}/{mask.sum()} correct)\n")
                else:
                    f.write(f"  {emotion:12s}: No test samples\n")
            
            f.write(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved: {summary_file}")


def main():
    """Main evaluation pipeline."""
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Create plots directory
    plots_dir = ROOT / 'Results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {plots_dir}\n")
    
    # Load dataset
    print("Loading TESS dataset...")
    data_root = ROOT / 'data'
    df = load_tess_dataset(str(data_root))
    train, val, test = get_splits(df)
    print(f"Using {len(test)} test samples\n")
    
    # Load models
    speech_model, text_model, fusion_model = load_models(device)
    
    # Load tokenizer for text models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Evaluate models
    print("\n" + "█"*60)
    print("STARTING MODEL EVALUATION")
    print("█"*60)
    
    # Use limited samples for faster evaluation (set to None for full dataset)
    max_samples = None  # Change to e.g., 100 for quick test
    
    speech_y_true, speech_y_pred, speech_acc = evaluate_speech_model(
        speech_model, test, device, max_samples=max_samples)
    
    text_y_true, text_y_pred, text_acc = evaluate_text_model(
        text_model, test, device, tokenizer, max_samples=max_samples)
    
    fusion_y_true, fusion_y_pred, fusion_acc = evaluate_fusion_model(
        fusion_model, test, device, tokenizer, max_samples=max_samples)
    
    # Save results
    print("\n" + "█"*60)
    print("SAVING RESULTS")
    print("█"*60 + "\n")
    
    results = {
        'speech': (speech_y_true, speech_y_pred),
        'text': (text_y_true, text_y_pred),
        'fusion': (fusion_y_true, fusion_y_pred),
    }
    
    # Save confusion matrices
    print("\nSaving Confusion Matrices:")
    save_confusion_matrix_plot(speech_y_true, speech_y_pred, 'Speech', str(plots_dir))
    save_confusion_matrix_plot(text_y_true, text_y_pred, 'Text', str(plots_dir))
    save_confusion_matrix_plot(fusion_y_true, fusion_y_pred, 'Fusion', str(plots_dir))
    
    # Save per-emotion accuracy plots
    print("\nSaving Per-Emotion Accuracy Plots:")
    save_accuracy_per_emotion_plot(speech_y_true, speech_y_pred, 'Speech', str(plots_dir))
    save_accuracy_per_emotion_plot(text_y_true, text_y_pred, 'Text', str(plots_dir))
    save_accuracy_per_emotion_plot(fusion_y_true, fusion_y_pred, 'Fusion', str(plots_dir))
    
    # Save accuracy tables
    print("\nSaving Accuracy Tables:")
    save_accuracy_table(speech_y_true, speech_y_pred, 'Speech', str(plots_dir))
    save_accuracy_table(text_y_true, text_y_pred, 'Text', str(plots_dir))
    save_accuracy_table(fusion_y_true, fusion_y_pred, 'Fusion', str(plots_dir))
    
    # Save comparison plots and matrices
    print("\nSaving Comparison Visualizations:")
    save_comparison_plot(results, str(plots_dir))
    accuracy_matrix_df = save_accuracy_matrix(results, str(plots_dir))
    
    # Save summary report
    print("\nSaving Summary Report:")
    save_summary_report(results, str(plots_dir))
    
    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nFinal Accuracies:")
    print(f"  Speech Model: {speech_acc:.4f}")
    print(f"  Text Model:   {text_acc:.4f}")
    print(f"  Fusion Model: {fusion_acc:.4f}")
    print(f"\nAll results saved to: {plots_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
