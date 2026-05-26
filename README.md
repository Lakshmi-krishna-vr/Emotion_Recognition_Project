# Multimodal Emotion Recognition using Speech, Text, and Fusion

A deep learning–based multimodal emotion recognition system that combines:

* **Speech-based emotion detection** using **BiLSTM + Attention**
* **Text-based emotion detection** using **Fine-tuned BERT**
* **Multimodal gated fusion** combining acoustic and linguistic representations

The project is trained and evaluated on the **TESS (Toronto Emotional Speech Set)** dataset and deployed as a real-time interactive **Streamlit web application**. 

---

# Live Demo

```txt
https://lakshmi-krishna-vr-emotion-recognition-project-app-zinsmb.streamlit.app/
```

---

# Project Overview

This project explores how emotional information can be extracted from:

1. Raw speech signals
2. Verbalized acoustic-prosodic descriptions
3. Combined multimodal representations

Three independent pipelines were trained and evaluated:

| Pipeline          | Input                | Model                |
| ----------------- | -------------------- | -------------------- |
| Speech-only       | MFCC audio features  | BiLSTM + Attention   |
| Text-only         | Prosody descriptions | Fine-tuned BERT      |
| Multimodal Fusion | Speech + Text        | Gated Fusion Network |

The system predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

---

# Dataset

## Toronto Emotional Speech Set (TESS)

The project uses the **TESS dataset**, containing recordings from two female actors speaking target words with different emotional tones. 

### Dataset Characteristics

* Clean studio-quality recordings
* WAV format
* 22,050 Hz mono audio
* 7 emotion classes
* Controlled emotional expressions

### Data Split

| Split      | Percentage |
| ---------- | ---------- |
| Training   | 70%        |
| Validation | 15%        |
| Test       | 15%        |

---

# System Architecture

## Overall Pipeline

```text
                Audio (.wav)
                      │
        ┌─────────────┴─────────────┐
        │                           │
   MFCC Extraction          Prosody Feature Extraction
        │                           │
 BiLSTM + Attention         Prosody Text Generation
        │                           │
 Speech Representation      Fine-tuned BERT
        │                           │
        └─────────────┬─────────────┘
                      │
               Gated Fusion
                      │
              Emotion Classifier
                      │
              7-class Emotion
```

---

# Speech Pipeline

## Feature Extraction using MFCCs

The speech pipeline processes raw audio into MFCC-based features. 

### Features Used

* 40 MFCC coefficients
* Delta coefficients (Δ)
* Delta-delta coefficients (ΔΔ)

Final feature dimension:

```text
40 × 3 = 120-dimensional feature vector
```

### Why MFCC?

MFCCs are highly effective for speech emotion recognition because they:

* Capture vocal timbre
* Preserve spectral information
* Encode speech dynamics
* Reduce irrelevant information

---

## BiLSTM Temporal Modeling

The extracted MFCC sequences are passed into a:

```text
2-layer Bidirectional LSTM
```

### Configuration

| Parameter            | Value          |
| -------------------- | -------------- |
| Hidden size          | 256            |
| Directions           | Bidirectional  |
| Final representation | 512 dimensions |
| Dropout              | 0.3            |

### Why BiLSTM?

BiLSTM captures:

* Past context
* Future context
* Emotional progression over time

This is important because emotions evolve throughout speech.

---

## Attention Mechanism

An attention layer learns which parts of speech are emotionally important.

Instead of treating all frames equally, attention focuses on:

* Sudden pitch changes
* Emotional bursts
* Intonation peaks
* Voice stress regions

This improves emotion discrimination significantly. 

---

# Text Pipeline

## Prosody-to-Text Conversion

Instead of using only raw transcripts, the system converts acoustic statistics into natural-language descriptions. 

### Extracted Prosody Features

The following acoustic properties are computed using `librosa`:

* RMS energy
* Pitch statistics (F0)
* Pitch variation
* Voiced ratio
* Zero-crossing rate
* Spectral centroid
* Harmonic-to-noise ratio

### Example Generated Description

```text
"The speaker has very loud and energetic voice,
high pitched tone, wide pitch variation,
fast speaking rate, bright sharp timbre."
```

---

## Fine-tuned BERT

The generated textual descriptions are processed using:

```text
bert-base-uncased
```

### Training Strategy

* Bottom 6 layers frozen
* Top 6 layers fine-tuned
* Max token length: 64

### Why BERT?

BERT understands semantic relationships between:

* Pitch descriptions
* Vocal intensity
* Speaking rate
* Voice quality

This allows emotion inference from verbalized acoustic patterns.

---

# Multimodal Fusion

## Gated Fusion Architecture

The speech and text embeddings are fused using a learnable gating mechanism. 

### Inputs

| Representation   | Dimension |
| ---------------- | --------- |
| Speech embedding | 512       |
| Text embedding   | 768       |

### Fusion Strategy

The gate dynamically decides:

* When to trust speech more
* When to trust text more

### Why Gated Fusion?

Simple concatenation treats all modalities equally.

Gated fusion learns adaptive weighting depending on the input characteristics.

Example:

* Angry → speech dominates
* Sad → text cues become more important

---

# Training Configuration

## Speech Model

| Parameter     | Value |
| ------------- | ----- |
| Optimizer     | AdamW |
| Learning Rate | 1e-3  |
| Batch Size    | 32    |
| Epochs        | 30    |

---

## Text Model

| Parameter     | Value |
| ------------- | ----- |
| Optimizer     | AdamW |
| Learning Rate | 2e-5  |
| Batch Size    | 32    |
| Epochs        | 15    |

---

## Fusion Model

| Parameter   | Value |
| ----------- | ----- |
| Non-BERT LR | 2e-5  |
| BERT LR     | 2e-6  |
| Batch Size  | 8     |
| Epochs      | 20    |

---

# Results

## Model Performance

| Pipeline          | Test Accuracy |
| ----------------- | ------------- |
| Speech-only       | 99.05%        |
| Text-only         | 59.52%        |
| Multimodal Fusion | 99.05%        |

---

# Streamlit Web Application

The project includes a fully interactive Streamlit deployment. 

## Features

* Upload `.wav` files
* Real-time emotion prediction
* Speech prediction
* Text prediction
* Fusion prediction
* Confidence score visualization
* Prosody text generation
* Comparative analysis across pipelines

---

# Tech Stack

## Languages

* Python

## Deep Learning Frameworks

* PyTorch
* Transformers

## Audio Processing

* Librosa
* NumPy

## NLP

* Hugging Face Transformers
* BERT

## Visualization

* Matplotlib
* Seaborn

## Deployment

* Streamlit
* Streamlit Community Cloud

---

# Project Structure

```text
emotion-recognition/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── speech_model.pth
│   ├── text_model.pth
│   └── fusion_model.pth
│
├── notebooks/
│   ├── speech_pipeline.ipynb
│   ├── text_pipeline.ipynb
│   └── fusion_pipeline.ipynb
│
├── utils/
│   ├── audio_utils.py
│   ├── text_utils.py
│   └── fusion_utils.py
│
├── dataset/
│   └── TESS/
│
└── assets/
    ├── architecture.png
    ├── confusion_matrix.png
    └── ui_demo.png
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Lakshmi-krishna-vr/Emotion_Recognition_Project/

cd emotion-recognition
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv emotion_env

emotion_env\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv emotion_env

source emotion_env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

# Dataset and Model Downloads

## Toronto Emotional Speech Set (TESS) Dataset

Download the dataset from the following Google Drive link:

```txt id="dataset-link"
https://drive.google.com/drive/folders/1RGzimPnIfldMy_3vvyCPFOguI0Gy3Mj_?usp=drive_link
```

---

# Pretrained Model Checkpoints

## Fusion Model

```txt id="fusion-model"
https://drive.google.com/file/d/14Nc9VKp3ALGyuaQQ1iaLt6o4LC0SNQtE/view?usp=drive_link
```

---

## Speech Model

```txt id="speech-model"
https://drive.google.com/file/d/1XG_4Oz8DzKRY_kz7OFdMlNT-nblsmz0X/view?usp=drive_link
```

---

## Text Model

```txt id="text-model"
https://drive.google.com/file/d/1kOPOc18u3gq8iESDpG1NkfgSsCTrxX0A/view?usp=drive_link
```

---

# Run the Application

```bash
streamlit run app.py
```

---
# Key Observations

## Speech Model Performance

The speech model achieved near-perfect accuracy because:

* TESS is clean studio audio
* Emotional acoustic patterns are well separated
* BiLSTM captures temporal dynamics effectively

---

## Text Model Limitations

The text-only pipeline underperformed because:

* TESS contains isolated words only
* Lexical information is weak
* Emotion mainly exists in vocal delivery

---

## Fusion Benefits

Fusion improved robustness for:

* Fear vs disgust
* Sad vs neutral
* Low-confidence speech samples

The gating layer dynamically shifts reliance between modalities. 

---

# Example Workflow

1. Upload a WAV audio file
2. System extracts MFCC features
3. Prosody statistics are computed
4. Prosody converted into natural language
5. Speech model predicts emotion
6. BERT predicts emotion from text
7. Fusion model combines both predictions
8. Final emotion displayed with confidence scores

---

# Future Improvements

Potential enhancements include:

* Transformer-based speech encoders
* Self-supervised speech models
* Real-world noisy datasets
* Emotion localization over time
* Better prosody verbalization
* Temporal-aware attention pooling
* Data augmentation techniques
* Real-time microphone inference



---

# Conclusion

This project demonstrates how multimodal learning can significantly improve emotion recognition systems by combining:

* Acoustic speech dynamics
* Linguistic reasoning
* Adaptive fusion mechanisms

The most important innovation is the **prosody verbalization bridge**, which enables a pre-trained language model like BERT to reason over acoustic emotion cues without modifying its architecture. 

---

# Author

## Lakshmi Krishna V R

* Saintgits College of Engineering
* Department of Computer Science and Engineering

---


