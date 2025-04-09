# Research & Selection

This section outlines three promising audio deepfake detection approaches from [Audio-Deepfake-Detection GitHub](https://github.com/media-sec-lab/Audio-Deepfake-Detection), evaluated for detecting AI-generated human speech, real-time potential, and real conversation analysis.

## 1. RawNet2
- **Key Technical Innovation**: End-to-end raw waveform processing with Sinc-Layers and GRU for feature extraction.
- **Reported Performance Metrics**: EER ~2-5% on ASVspoof 2019 LA; ~10-15% on unseen datasets (e.g., WaveFake).
- **Why Promising**: Minimal preprocessing supports near real-time detection; effective for AI-generated speech artifacts.
- **Limitations**: Limited generalization to noisy real-world data; computationally intensive.

## 2. RawBMamba
- **Key Technical Innovation**: Bidirectional state-space model (SSM) for efficient long-range dependency modeling in raw audio.
- **Reported Performance Metrics**: EER ~1-3% on ASVspoof 2021, outperforming RawNet2.
- **Why Promising**: Efficient processing and strong speech dynamics modeling suit real-time conversation analysis.
- **Limitations**: Newer model with less real-world testing; complex implementation.

## 3. Prosodic and Pronunciation Features (LCNN + Bi-LSTM)
- **Key Technical Innovation**: Combines prosodic (pitch, energy) and pronunciation features with LCNN and Bi-LSTM, enhanced by attention and wav2vec embeddings.
- **Reported Performance Metrics**: EER ~10-15% on ASVspoof 2015, with 5-10% improvement over baselines.
- **Why Promising**: Speech-specific features target AI-generated anomalies; lightweight LCNN suits near real-time use.
- **Limitations**: Preprocessing overhead (e.g., phoneme recognition); sensitive to noise.

## Selection Rationale
These models balance innovation, performance, and applicability to Momenta’s goals. I chose the Prosodic + Pronunciation approach for implementation due to its interpretability and prior success with a similar LCNN-BiLSTM setup (see `implementation.ipynb`).
