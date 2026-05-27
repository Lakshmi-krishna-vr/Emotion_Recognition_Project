=========================================================================
PROJECT: MULTIMODAL EMOTION RECOGNITION SYSTEM
Saintgits College of Engineering
=========================================================================

SUBMITTED BY:
1. Lakshmi Krishna V R   (Roll No: MGP23CS088)

DATE: May 27, 2026


=========================================================================
1. PROJECT LINKS
=========================================================================

Please copy and paste these links into your browser to access the project.

[A] GITHUB REPOSITORY (Source Code & Documentation)

https://github.com/Lakshmi-krishna-vr/Emotion_Recognition_Project


[B] LIVE STREAMLIT WEB APPLICATION

https://lakshmi-krishna-vr-emotion-recognition-project-app-zinsmb.streamlit.app/


[C] DATASET DOWNLOAD LINK (TESS DATASET)

https://drive.google.com/drive/folders/1RGzimPnIfldMy_3vvyCPFOguI0Gy3Mj_?usp=drive_link


[D] PRETRAINED MODEL CHECKPOINTS

Fusion Model:
https://drive.google.com/file/d/14Nc9VKp3ALGyuaQQ1iaLt6o4LC0SNQtE/view?usp=drive_link

Speech Model:
https://drive.google.com/file/d/1XG_4Oz8DzKRY_kz7OFdMlNT-nblsmz0X/view?usp=drive_link

Text Model:
https://drive.google.com/file/d/1kOPOc18u3gq8iESDpG1NkfgSsCTrxX0A/view?usp=drive_link



=========================================================================
2. PROJECT ABSTRACT
=========================================================================

This project implements a Multimodal Emotion Recognition System capable
of identifying human emotions from speech signals using three independent
deep learning pipelines:

1. Speech-Only Emotion Recognition
2. Text-Only Emotion Recognition
3. Multimodal Gated Fusion Emotion Recognition

The system combines acoustic speech analysis and natural language
understanding to improve emotion classification robustness.

The implemented architecture includes:

1. MFCC (Mel Frequency Cepstral Coefficients)
   - Extracts spectral acoustic information from speech.

2. BiLSTM + Attention
   - Learns temporal emotional patterns from MFCC sequences.

3. BERT (Bidirectional Encoder Representations from Transformers)
   - Processes verbalized prosody descriptions.

4. Gated Multimodal Fusion
   - Dynamically combines speech and text embeddings.

The system classifies emotions into 7 categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise
- Sad

The project is trained and evaluated on the Toronto Emotional Speech
Set (TESS) dataset.



=========================================================================
3. DATASET INFORMATION
=========================================================================

Dataset Used:
Toronto Emotional Speech Set (TESS)

Dataset Characteristics:
- Studio-quality emotional speech recordings
- WAV audio format
- Sampling rate: 22,050 Hz
- Clean and noise-free recordings
- Two female speakers
- Seven emotion classes

Dataset Split:
- Training: 70%
- Validation: 15%
- Test: 15%

Target Emotion Classes:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Pleasant Surprise
7. Sad



=========================================================================
4. SYSTEM REQUIREMENTS & DEPENDENCIES
=========================================================================

Recommended Environment:
- Python 3.10 or above
- Google Colab / Jupyter Notebook
- VS Code Recommended

Pretrained Transformer Model:
- bert-base-uncased

Install the required Python libraries using:

pip install torch transformers librosa streamlit scikit-learn \
numpy pandas matplotlib seaborn soundfile

DEPENDENCY DETAILS

1. torch
   - Deep learning backend for model training and inference.

2. transformers
   - Loads pretrained BERT language model.

3. librosa
   - Audio preprocessing and MFCC extraction.

4. streamlit
   - Interactive web application deployment.

5. scikit-learn
   - Metrics, preprocessing, and evaluation.

6. numpy
   - Numerical array operations.

7. pandas
   - Data handling and CSV management.

8. matplotlib
   - Visualization and training plots.

9. seaborn
   - Confusion matrix visualization.

10. soundfile
    - WAV file processing support.



=========================================================================
5. PROJECT WORKFLOW
=========================================================================

STEP 1 — ENVIRONMENT SETUP
---------------------------------
- Install all required libraries.
- Download the TESS dataset.
- Download pretrained model checkpoints.


STEP 2 — AUDIO PREPROCESSING
---------------------------------
- Load WAV audio files using librosa.
- Normalize audio signals.
- Pad or truncate audio to fixed length.
- Extract MFCC features.


STEP 3 — FEATURE EXTRACTION
---------------------------------

[A] Speech Features
    - MFCC coefficients
    - Delta coefficients
    - Delta-Delta coefficients

[B] Prosody Features
    - RMS Energy
    - Pitch statistics
    - Spectral centroid
    - Harmonic-to-noise ratio
    - Speaking rate

[C] Prosody-to-Text Conversion
    - Acoustic statistics converted into natural-language descriptions.


STEP 4 — MODEL TRAINING
---------------------------------

[A] Speech Pipeline
    - BiLSTM + Attention trained on MFCC features.

[B] Text Pipeline
    - Fine-tuned BERT trained on generated prosody descriptions.

[C] Fusion Pipeline
    - Gated fusion combines speech and text embeddings.


STEP 5 — INFERENCE & PREDICTION
---------------------------------
- User uploads a WAV audio file.
- Speech model predicts emotion.
- Text model predicts emotion.
- Fusion model combines both predictions.
- Confidence scores are displayed in real time.



=========================================================================
6. MODEL DETAILS
=========================================================================

Speech Model:
- Architecture: BiLSTM + Attention
- Input: MFCC Feature Sequences
- Output: 7 Emotion Classes

Text Model:
- Architecture: Fine-tuned BERT
- Input: Prosody Description Text
- Output: 7 Emotion Classes

Fusion Model:
- Architecture: Gated Multimodal Fusion
- Input: Speech + Text Embeddings
- Output: Final Emotion Prediction


=========================================================================
7. PERFORMANCE RESULTS
=========================================================================

Pipeline Performance:

1. Speech-Only Model
   - Accuracy: 99.05%

2. Text-Only Model
   - Accuracy: 59.52%

3. Multimodal Fusion Model
   - Accuracy: 99.05%

Key Observations:
- Speech pipeline performs extremely well on clean TESS audio.
- Text-only pipeline struggles because TESS contains isolated words.
- Fusion improves robustness for ambiguous emotional cases.



=========================================================================
8. HOW TO RUN THE PROJECT
=========================================================================

STEP 1:
Clone the repository

git clone https://github.com/Lakshmi-krishna-vr/Emotion_Recognition_Project.git


STEP 2:
Install dependencies

pip install -r requirements.txt

STEP 3:
Download dataset and model checkpoints from the provided links.


STEP 4:
Create Virtual Environment
Windows

python -m venv emotion_env

emotion_env\Scripts\activate


STEP 5:
Run the Streamlit application

streamlit run app.py


STEP 6:
Upload a WAV audio sample and view:
- Speech prediction
- Text prediction
- Fusion prediction
- Confidence graphs


=========================================================================
9. PROJECT STRUCTURE
=========================================================================

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




=========================================================================
10. DEPLOYMENT
=========================================================================

The project is deployed using Streamlit Community Cloud.

Deployment Features:
- Browser-based interface
- WAV audio upload
- Real-time emotion prediction
- Confidence score visualization
- Multimodal analysis dashboard

Deployment Platform:
Streamlit Community Cloud



=========================================================================
11. FUTURE IMPROVEMENTS
=========================================================================

Potential future enhancements include:

- Real-time microphone recording
- Transformer-based speech encoders
- Self-supervised audio models
- Real-world noisy speech datasets
- Mobile application deployment
- Emotion timeline visualization
- Multilingual emotion recognition
- Better prosody verbalization methods



=========================================================================
12. CONCLUSION
=========================================================================

The Multimodal Emotion Recognition System demonstrates how acoustic
speech analysis and transformer-based language understanding can be
combined to build robust emotion classification systems.

The project highlights the effectiveness of:
- MFCC-based acoustic representations
- BiLSTM temporal modeling
- Attention mechanisms
- BERT-based contextual understanding
- Gated multimodal fusion

A major contribution of this work is the prosody verbalization bridge,
which enables BERT to reason over acoustic speech patterns through
natural language descriptions without modifying the transformer
architecture.

=========================================================================