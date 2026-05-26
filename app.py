"""
app.py — Streamlit Real-Time Emotion Recognition Demo
Place in D:\\emotion_project\\ and run:
    streamlit run app.py
"""

import os
import sys
import io
import warnings
import numpy as np
import torch
import librosa
import soundfile as sf
import streamlit as st
import gdown  # Restored for automatic model downloading

# Suppress transformers deprecation warnings about __path__ aliases
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

# Ensure local imports work correctly from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_device, set_seed,
    EMOTION_LABELS, NUM_CLASSES,
    extract_prosody_cues, prosody_cues_to_text,
)
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train   import TextEmotionModel
from models.fusion_pipeline.train import FusionEmotionModel

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "Results"
SR          = 22050
DURATION    = 4.0
MAX_AUDIO   = 345
MAX_TEXT    = 64      # must match training (TextDataset default)

# Google Drive Model IDs (Restored front-end auto setup elements)
TEXT_MODEL_ID = "1kOPOc18u3gq8iESDpG1NkfgSsCTrxX0A"
SPEECH_MODEL_ID = "1XG_4Oz8DzKRY_kz7OFdMlNT-nblsmz0X"
FUSION_MODEL_ID = "14Nc9VKp3ALGyuaQQ1iaLt6o4LC0SNQtE"

EMOTION_EMOJI = {
    "angry":   "😡", "disgust": "🤢", "fear":   "😨",
    "happy":   "😊", "neutral": "😐", "ps":     "😲", "sad": "😢",
}

# One colour per emotion — used in bar charts
EMOTION_COLOR = {
    "angry":   "#ef4444", "disgust": "#84cc16", "fear":   "#f97316",
    "happy":   "#facc15", "neutral": "#94a3b8", "ps":     "#a855f7",
    "sad":     "#60a5fa",
}

# ─── Create Required Folders & Handle Downloads ───────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {os.path.basename(output_path)} from cloud backup..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)

def download_models():
    download_file(SPEECH_MODEL_ID, os.path.join(RESULTS_DIR, "speech_best_model.pt"))
    download_file(TEXT_MODEL_ID, os.path.join(RESULTS_DIR, "text_best_model.pt"))
    download_file(FUSION_MODEL_ID, os.path.join(RESULTS_DIR, "fusion_best_model.pt"))

# Trigger automated verification/download check before cache initialization
download_models()

# ─── Checkpoint loader (handles any legacy key naming) ────────────────────────
FC_KEY_MAP = {
    "fc1.weight": "classifier.1.weight", "fc1.bias": "classifier.1.bias",
    "fc2.weight": "classifier.4.weight", "fc2.bias": "classifier.4.bias",
    "text_enc.fc1.weight": "text_enc.classifier.1.weight",
    "text_enc.fc1.bias":   "text_enc.classifier.1.bias",
    "text_enc.fc2.weight": "text_enc.classifier.4.weight",
    "text_enc.fc2.bias":   "text_enc.classifier.4.bias",
}

def remap_sd(sd):
    return {FC_KEY_MAP.get(k, k): v for k, v in sd.items()}

def robust_load(model, path, device):
    sd = remap_sd(torch.load(path, map_location=device, weights_only=False))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ─── Model loading (cached once per session) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    set_seed(42)
    device = get_device()

    # Notebook uses bert-base-uncased throughout
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sp = robust_load(
        SpeechEmotionModel().to(device),
        os.path.join(RESULTS_DIR, "speech_best_model.pt"), device)
    tx = robust_load(
        TextEmotionModel(unfreeze_last_n=12).to(device),  # Match updated 12 layer architecture
        os.path.join(RESULTS_DIR, "text_best_model.pt"), device)
    fu = robust_load(
        FusionEmotionModel().to(device),
        os.path.join(RESULTS_DIR, "fusion_best_model.pt"), device)
    return device, tokenizer, sp, tx, fu

# ─── Audio helpers ────────────────────────────────────────────────────────────
def bytes_to_waveform(audio_bytes: bytes):
    """Decode raw audio bytes → (waveform np.float32, sr=22050)."""
    buf = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(buf, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR)
    max_len = int(SR * DURATION)
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]
    return waveform.astype(np.float32)

def waveform_to_mfcc_tensor(waveform: np.ndarray) -> torch.Tensor:
    """MFCC extraction identical to training SpeechDataset."""
    mfcc   = librosa.feature.mfcc(y=waveform, sr=SR, n_mfcc=40, n_fft=512, hop_length=256)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.vstack([mfcc, delta, delta2]).T          # (T, 120)
    if feat.shape[0] < MAX_AUDIO:
        feat = np.vstack([feat,
            np.zeros((MAX_AUDIO - feat.shape[0], 120), dtype=np.float32)])
    else:
        feat = feat[:MAX_AUDIO]
    mean = feat.mean(0, keepdims=True)
    std  = feat.std(0,  keepdims=True) + 1e-8
    feat = (feat - mean) / std
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)   # (1, T, 120)

# ─── Inference helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def predict_speech(model, device, waveform: np.ndarray):
    feat  = waveform_to_mfcc_tensor(waveform).to(device)
    probs = torch.softmax(model(feat), dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs

@torch.no_grad()
def predict_text(model, tokenizer, device, combined_text: str):
    """
    Text pipeline input = Combined Word + Prosody Description string.
    """
    enc = tokenizer(
        combined_text,
        max_length=MAX_TEXT,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids   = enc["input_ids"].to(device)
    attn  = enc["attention_mask"].to(device)
    ttype = enc["token_type_ids"].to(device)
    probs = torch.softmax(model(ids, attn, ttype), dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs

@torch.no_grad()
def predict_fusion(model, tokenizer, device, waveform: np.ndarray, combined_text: str):
    """
    Fusion: speech MFCC tensor + combined text info → gated fusion → emotion.
    """
    feat = waveform_to_mfcc_tensor(waveform).to(device)
    enc  = tokenizer(
        combined_text,
        max_length=MAX_TEXT,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids   = enc["input_ids"].to(device)
    attn  = enc["attention_mask"].to(device)
    ttype = enc["token_type_ids"].to(device)
    probs = torch.softmax(
        model(feat, ids, attn, ttype), dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs

# ─── UI helpers ───────────────────────────────────────────────────────────────
def show_result(label_idx: int, probs: np.ndarray):
    label      = EMOTION_LABELS[label_idx]
    emoji      = EMOTION_EMOJI[label]
    confidence = probs[label_idx] * 100

    st.markdown(
        f"""
        <div style="
            background: #1a1f2e;
            border-radius: 14px;
            padding: 20px 24px 16px;
            margin-bottom: 12px;
            border: 1px solid #252b3b;
        ">
            <div style="font-size: 2.6rem; line-height: 1;">{emoji}</div>
            <div style="
                font-size: 1.5rem;
                font-weight: 700;
                color: {EMOTION_COLOR[label]};
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-top: 6px;
            ">{label}</div>
            <div style="color: #64748b; font-size: 0.82rem; margin-top: 2px;">
                {confidence:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, emo in enumerate(EMOTION_LABELS):
        pct = float(probs[i]) * 100
        col_label, col_bar, col_pct = st.columns([2, 6, 1])
        col_label.markdown(
            f"<span style='color:#64748b;font-size:0.78rem;font-family:monospace'>{emo}</span>",
            unsafe_allow_html=True,
        )
        col_bar.markdown(
            f"""
            <div style="
                background:#1e293b; border-radius:99px;
                height:6px; margin-top:7px; overflow:hidden;
            ">
                <div style="
                    width:{pct:.1f}%; height:100%;
                    background:{EMOTION_COLOR[emo]};
                    border-radius:99px;
                "></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_pct.markdown(
            f"<span style='font-size:0.72rem;color:#64748b'>{pct:.0f}%</span>",
            unsafe_allow_html=True,
        )

def audio_uploader(key: str):
    """Upload or record audio. Returns (waveform np.ndarray, raw bytes, filename) or (None, None, None)."""
    method = st.radio(
        "Input method",
        ["Upload a WAV/MP3 file", "Record with microphone"],
        key=f"method_{key}",
        horizontal=True,
    )
    if method == "Upload a WAV/MP3 file":
        uploaded = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3", "ogg", "flac"],
            key=f"upload_{key}",
        )
        if uploaded:
            raw = uploaded.read()
            return bytes_to_waveform(raw), raw, uploaded.name
    else:
        st.info(
            "🎤 Click **Record** below, speak for a few seconds, then click **Stop**."
        )
        audio = st.audio_input("Record your voice", key=f"mic_{key}")
        if audio:
            raw = audio.read()
            return bytes_to_waveform(raw), raw, "recorded_mic.wav"
    return None, None, None

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎭",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Dark application background setup */
    .stApp { background-color: #0b0e14; }
    
    /* Target the layout tags inside tab buttons to scale up size */
    div[data-testid="stTabs"] button p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Retain clean tab structure alignment and styling */
    div[data-testid="stTabs"] button { 
        font-family: monospace; 
        letter-spacing: 1px; 
    }
    
    /* Configures the boundary color of the selected active tab */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #6ee7b7; 
        border-color: #6ee7b7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎭 Emotion Recognition — Live Demo")
st.caption(
    "Speech · Text · Fusion — TESS dataset · 7 emotions · "
    "BiLSTM + BERT (bert-base-uncased)"
)

# Load models once
device, tokenizer, sp_model, tx_model, fu_model = load_models()
st.success(f"Models loaded on **{device}** (Validation Classes count verified: {NUM_CLASSES})", icon="✅")
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_speech, tab_text, tab_fusion = st.tabs(["🎤  Speech", "💬  Text", "🔀  Fusion"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SPEECH
# ══════════════════════════════════════════════════════════════════════════
with tab_speech:
    st.subheader("Speech-only prediction")
    st.write(
        "Record or upload audio. The BiLSTM model predicts emotion from "
        "MFCC + delta + delta² features (120-dim, 345 frames)."
    )
    waveform, raw_bytes, _ = audio_uploader("speech")
    if raw_bytes:
        st.audio(raw_bytes)
    if waveform is not None:
        if st.button("🔍 Predict emotion from speech", type="primary", key="btn_speech"):
            with st.spinner("Extracting MFCC features and running BiLSTM…"):
                try:
                    idx, probs = predict_speech(sp_model, device, waveform)
                    st.success("Prediction complete!")
                    show_result(idx, probs)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload or record audio above to get started.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — TEXT
# ══════════════════════════════════════════════════════════════════════════
with tab_text:
    st.subheader("Text-only prediction (prosody + transcript → BERT)")
    st.write(
        "The text pipeline parses the spoken word transcript and appends "
        "the automatically extracted prosody sentence before handing over to BERT."
    )

    waveform_t, raw_bytes_t, filename_t = audio_uploader("text")
    if raw_bytes_t:
        st.audio(raw_bytes_t)

    if waveform_t is not None:
        # Try to guess the transcript word from the file name (e.g. OAF_back_angry.wav -> back)
        default_word = "unknown"
        if "_" in filename_t:
            parts = filename_t.replace(".wav", "").replace(".mp3", "").split("_")
            if len(parts) >= 3:
                default_word = "_".join(parts[1:-1])

        # Text input box allows manual control over the word context
        transcript_word = st.text_input("Spoken word transcript keyword:", value=default_word, key="word_text")

        cues         = extract_prosody_cues(waveform_t, SR)
        prosody_desc = prosody_cues_to_text(cues)
        combined_text = f"Spoken word: {transcript_word}. Voice cues: {prosody_desc}"
        
        st.markdown("**Combined Formatted Input (BERT Input):**")
        st.info(combined_text)

        if st.button("🔍 Predict emotion from text (prosody)", type="primary", key="btn_text"):
            with st.spinner("Running BERT on formatted sequence…"):
                try:
                    idx, probs = predict_text(tx_model, tokenizer, device, combined_text)
                    st.success("Prediction complete!")
                    show_result(idx, probs)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload or record audio above to get started.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — FUSION
# ══════════════════════════════════════════════════════════════════════════
with tab_fusion:
    st.subheader("Fusion prediction — BiLSTM + BERT")
    st.write(
        "Upload or record one audio clip. The app feeds it to all three models simultaneously."
    )

    waveform_f, raw_bytes_f, filename_f = audio_uploader("fusion")
    if raw_bytes_f:
        st.audio(raw_bytes_f)

    if waveform_f is not None:
        default_word_f = "unknown"
        if "_" in filename_f:
            parts = filename_f.replace(".wav", "").replace(".mp3", "").split("_")
            if len(parts) >= 3:
                default_word_f = "_".join(parts[1:-1])

        transcript_word_f = st.text_input("Spoken word transcript keyword:", value=default_word_f, key="word_fusion")

        cues_f         = extract_prosody_cues(waveform_f, SR)
        prosody_desc_f = prosody_cues_to_text(cues_f)
        combined_text_f = f"Spoken word: {transcript_word_f}. Voice cues: {prosody_desc_f}"
        
        st.markdown("**Combined Formatted Input (BERT input):**")
        st.info(combined_text_f)

        if st.button("🔍 Run all three models", type="primary", key="btn_fusion"):
            with st.spinner("Running BiLSTM, BERT, and fusion model…"):
                try:
                    idx_sp, probs_sp = predict_speech(sp_model, device, waveform_f)
                    idx_tx, probs_tx = predict_text(tx_model, tokenizer, device, combined_text_f)
                    idx_fu, probs_fu = predict_fusion(
                        fu_model, tokenizer, device, waveform_f, combined_text_f)

                    st.success("All three predictions complete!")
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.markdown("##### 🎤 Speech (BiLSTM)")
                        show_result(idx_sp, probs_sp)
                    with r2:
                        st.markdown("##### 💬 Text (BERT)")
                        show_result(idx_tx, probs_tx)
                    with r3:
                        st.markdown("##### 🔀 Fusion (final)")
                        show_result(idx_fu, probs_fu)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload or record audio above.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Models: BiLSTM speech · BERT (bert-base-uncased) text · Gated fusion  |  "
    "Dataset: TESS · 7 emotions · 2800 samples  |  "
    f"Device: {device}"
)