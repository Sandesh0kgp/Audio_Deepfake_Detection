
# Audio Deepfake Detection

This repository contains my solution for the Momenta Audio Deepfake Detection Take-Home Assessment. It implements a hybrid LCNN-BiLSTM model with prosodic features to detect AI-generated audio, achieving 89.50% accuracy on ASVspoof 2019 LA data.

## Setup
1. Clone the repo: `git clone github.com/yourusername/audio-deepfake-detection`
2. Install dependencies: `pip install -r requirements.txt`
3. Download ASVspoof 2019 LA dataset and place in `/kaggle/input/asvspoof-2019-dataset/LA/` (or adjust paths).

## Usage
- **Train**: Run `implementation.py` in Jupyter or Kaggle.
- **Test**: Use `best_model.pth` with the test function in the notebook.

## Results
- Best Eval Accuracy: 89.50% (5 epochs).
- See `research.md` for model selection, `implementation.py` for code, and `analysis.md` for details.

## License
MIT License
