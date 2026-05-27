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

# Project Structure

```text
 emotion_project/
  ├── utils.py
  ├── local_test.py
  ├── analyze.py
  ├── requirements.txt
  ├── VSCODE_SETUP.sh
  ├── Emotion_Recognition_Training.ipynb
  ├── .vscode/
  │   ├── launch.json
  │   └── settings.json
  ├── models/
  │   ├── __init__.py
  │   ├── speech_pipeline/
  │   │   ├── __init__.py
  │   │   ├── train.py
  │   │   └── test.py
  │   ├── text_pipeline/
  │   │   ├── __init__.py
  │   │   ├── train.py
  │   │   └── test.py
  │   └── fusion_pipeline/
  │       ├── __init__.py
  │       ├── train.py
  │       └── test.py
  ├── data/                      ← dataset goes here (Step H)
  └── Results/
          |-prosody_descriptions.csv          ← models from Colab go here (Step 3)

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

Download dataset and model checkpoints from the provided links.

---

# Run the Application

```bash
streamlit run app.py
```

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


