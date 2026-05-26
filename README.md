# Multimodal Emotion Recognition using Gated Fusion on TESS

[![Streamlit App](https://static.streamlit.io/badge.svg)](https://lakshmi-krishna-vr-emotion-recognition-project-app-zinsmb.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)

An end-to-end deep learning framework that implements a multi-stage **Gated Multimodal Fusion Architecture** to perform 7-class emotion recognition on the **Toronto Emotional Speech Set (TESS)**. This system balances two independent sensory contexts: raw audio temporal dynamics and rule-based acoustic-prosodic feature verbalizations analyzed using a fine-tuned Pre-trained Language Model (BERT).

---

## 🚀 Live Deployment & Web Application
The application is fully optimized, compiled, and deployed in production. You can interact with the live application, upload audio files, and verify inference streams directly through the web browser:

👉 **[Live Streamlit Application Portal](https://lakshmi-krishna-vr-emotion-recognition-project-app-zinsmb.streamlit.app/)**

### 🖥️ Production Web UI Preview
When an audio track is supplied to the application interface, the cloud infrastructure dynamically extracts acoustic statistics, verbalizes the prosody profile, and displays side-by-side classification confidence distributions for all three deep learning blocks.

<p align="center">
  <img src="image_c0313f.png" alt="Streamlit Web Application Dashboard Interface" width="90%" />
</p>

---

## ⚙️ Architecture & Core Components

This project splits processing into three concurrent pipelines using standard audio boundaries:

1. **Speech-Only Pipeline:** Extracts 40-dimensional Mel-Frequency Cepstral Coefficients (MFCC) alongside their first and second-order temporal derivatives ($\Delta$ and $\Delta\Delta$), yielding a 120-dimensional frame feature vector. These inputs pass through a 2-layer **Bidirectional LSTM (BiLSTM)** supported by an **Additive Single-Layer Attention Mechanism** to emphasize high-confidence emotional fragments.
2. **Text-Only Pipeline:** Computes 9 structural prosodic statistics (RMS energy, fundamental frequency $F_0$, zero-crossing rates, spectral centroids, and Harmonic-to-Noise Ratios) via `librosa`. These statistics are translated into natural-language sentences (e.g., *"The speaker has high pitched tone, fast speaking rate..."*) and combined with the target token, bypassing structural limitations using a parameter-efficient **Fine-Tuned BERT Encoder** (top 6 layers unfrozen).
3. **Gated Multimodal Fusion Pipeline:** Integrates hidden contexts from both architectures. Instead of uniform feature concatenation,
4. it evaluates a learnable **Gated Fusion Network** that dynamically assigns importance weights to each sensory branch based on dimensional proximity matrixes.

┌───────────────┐
               │  Audio File   │
               └───────┬───────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[MFCC Extraction]             [Prosody Verbalization]
Shape: [T, 120]               "Voice cues: Loud voice..."
│                               │
▼                               ▼
BiLSTM + Attention              Fine-Tuned BERT
Speech Context vector (s)      Text Context vector (t)
│                               │
└───────────────┬───────────────┘
▼
┌──────────────────┐
│   Gated Fusion   │◄── Dynamically scales weights
└──────────────────┘
│
▼
┌──────────────────┐
│  7-Class Output  │
└──────────────────┘
---

## 📊 Experimental Setup & Performance

Models were trained utilizing a stratified split (70% Train, 15% Val, 15% Test) on the TESS dataset across 7 core emotion categories: *Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad*.

### Benchmark Evaluation Metrics

| Pipeline Framework | Modality Vector Input | Core Model Architecture | Test Split Accuracy | Training Window |
| :--- | :--- | :--- | :---: | :---: |
| **Speech-Only** | Audio (MFCC + $\Delta$ + $\Delta\Delta$) | BiLSTM + Attention | **99.05%** | 30 Epochs |
| **Text-Only** | Verbalized Acoustic Heuristics | BERT-Base (Top-6 Fine-tuned) | **59.52%** | 15 Epochs |
| **Multimodal Fusion** | Integrated Embedded Latents | Gated Cross-Modal Networks | **99.05%** | 20 Epochs |

> 📌 **Key Architectural Insight:** The isolated word metrics for the Text-Only setup ($59.52\%$) reflect the strict linguistic constraints of utilizing single-word transcripts ("dog", "bird"). However, when context features overlap or boundary ambiguities arise within raw audio (e.g., *Disgust vs. Fear*), the **Gated Fusion layer** successfully balances channels to resolve confusion matrices effectively, allowing the multi-modal network to anchor securely to the high-performing speech dynamics channel ($99.05\%$).

---

## 📂 Repository Structure
├── Results/                        # Saved checkpoint metrics and analytics
│   ├── speech_best_model.pt        # Trained BiLSTM Attention weights (9.9 MB)
│   ├── text_best_model.pt          # Fine-tuned BERT checkpoint weights (438.8 MB)
│   ├── fusion_best_model.pt        # Trained Gated Fusion framework weights (453.2 MB)
│   ├── speech_history.png          # Optimization history profiles for speech
│   ├── speech_confusion_matrix.png # Final speech class performance matrix
│   ├── text_history.png

│   ├── text_confusion_matrix.png

│   ├── fusion_history.png

│   └── fusion_confusion_matrix.png # Unified gating validation metrics
├── models/                         # Training logic routines
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
├── image_c0313f.png                # Streamlit live app user interface preview image
├── app.py                          # Streamlit application orchestration file
├── local_test.py                   # Command-line smoke test inference module
├── utils.py                        # Librosa audio features and verbalization bridges
└── requirements.txt                # Operational environment pinning requirements

## 🛠️ Local Installation & Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/multimodal-emotion-recognition.git](https://github.com/your-username/multimodal-emotion-recognition.git)
   cd multimodal-emotion-recognition

2. Configure Environment Dependencies:
Ensure you have Python 3.10 or higher installed. Install dependencies using:

Bash
pip install -r requirements.txt
Note: System-level audio manipulation may require ffmpeg installed on your host architecture.

💻 Usage & Verification
Run Automated Local Evaluation (Smoke-Test)
You can evaluate any arbitrary .wav audio track across all three pipeline states using the command-line utility script local_test.py:

python local_test.py \
    --audio path/to/sample.wav \
    --speech_ckpt Results/speech_best_model.pt \
    --text_ckpt   Results/text_best_model.pt \
    --fusion_ckpt Results/fusion_best_model.pt

Launch Interactive Streamlit Server Locally
To boot up the complete visual user interface along with local device acceleration hosting, execute:

streamlit run app.py

📈 Model Training History Logs
Below are the objective loss metrics and normalized validation matrices captured directly from the experimental environments:

1. Speech Architecture Analysis
Smooth validation descent down to cross-entropy ceilings with near-total class differentiation.

2. Linguistic Text Architecture Analysis
Highlights natural performance limitations caused by single-word lexical constraints.

3. Gated Multimodal Fusion Analysis
Demonstrates rapid optimization stability and total structural recovery across all classifications.

🎓 Acknowledgments & References
Dataset: Toronto Emotional Speech Set (TESS) supplied by the University of Toronto, Department of Psychology.

Developed as part of an academic deep-learning exploration into cross-modal integration bridges using PyTorch and Hugging Face Transformers.
