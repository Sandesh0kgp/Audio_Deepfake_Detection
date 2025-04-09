# Implementation - ProsodicLCNNBiLSTM for Audio Deepfake Detection
# This script implements a hybrid LCNN-BiLSTM model with prosodic features for Momenta's Audio Deepfake Detection task.
# Based on the 'Detection Using Prosodic and Pronunciation Features' approach, adapted for ASVspoof 2019 LA dataset.

import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Feature Extraction
class ProsodicFeatureExtractor(nn.Module):
    """Extracts pitch and energy features from audio waveforms."""
    def __init__(self):
        super().__init__()

    def forward(self, waveform):
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate=16000)  # [batch, ~399]
        energy = waveform ** 2  # [batch, samples]
        energy = torch.mean(energy.view(energy.size(0), -1, 160), dim=2)  # [batch, ~399]
        # Match CNN output time steps (50)
        pitch = torch.nn.functional.interpolate(pitch.unsqueeze(1), size=50, mode='linear', align_corners=False).squeeze(1)
        energy = torch.nn.functional.interpolate(energy.unsqueeze(1), size=50, mode='linear', align_corners=False).squeeze(1)
        return torch.stack([pitch, energy], dim=2)  # [batch, 50, 2]

# Hybrid Model
class ProsodicLCNNBiLSTM(nn.Module):
    """Hybrid LCNN-BiLSTM model for audio spoof detection."""
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        self.lstm = nn.LSTM(
            input_size=128*8 + 2,  # 1026 features
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64
        )
        self.prosodic_extractor = ProsodicFeatureExtractor()

    def forward(self, x):
        mel = torch.log(self.mel_spec(x) + 1e-6).unsqueeze(1)  # [batch, 1, 64, ~399]
        cnn_out = self.cnn(mel)  # [batch, 128, 8, 50]
        cnn_out = cnn_out.permute(0, 3, 1, 2)  # [batch, 50, 128, 8]
        cnn_out = cnn_out.reshape(cnn_out.size(0), cnn_out.size(1), -1)  # [batch, 50, 1024]
        
        prosodic = self.prosodic_extractor(x)  # [batch, 50, 2]
        
        combined = torch.cat([cnn_out, prosodic], dim=2)  # [batch, 50, 1026]
        lstm_out, _ = self.lstm(combined)
        return self.classifier(lstm_out[:, -1, :])

# Dataset Class
class ASVSpoofDataset(Dataset):
    """Dataset class for ASVspoof 2019 audio files."""
    def __init__(self, file_list, label_list, max_length=64000):
        self.files = file_list
        self.labels = label_list
        self.max_length = max_length
        print(f"Loaded {len(self.files)} samples")
        print(f"Class distribution: {Counter(self.labels)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.files[idx])
        audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.mean(dim=0)
        
        if random.random() > 0.5:
            noise = torch.randn_like(audio) * 0.005  # Light noise augmentation
            audio = audio + noise
        
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio, self.labels[idx]

# Training Function
def train_model():
    """Train the ProsodicLCNNBiLSTM model on ASVspoof 2019 data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths (adjust if not using Kaggle)
    train_dir = "/kaggle/input/asvspoof-2019-dataset/LA/ASVspoof2019_LA_train/flac"
    eval_dir = "/kaggle/input/asvspoof-2019-dataset/LA/ASVspoof2019_LA_eval/flac"
    
    all_files = []
    all_labels = []
    
    for f in os.listdir(train_dir):
        if f.endswith('.flac'):
            all_files.append(os.path.join(train_dir, f))
            all_labels.append(1 if "LA_T_" in f else 0)
    
    for f in os.listdir(eval_dir):
        if f.endswith('.flac'):
            all_files.append(os.path.join(eval_dir, f))
            all_labels.append(1 if "LA_T_" in f else 0)
    
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    
    split_idx = int(0.8 * len(all_files))
    train_files, train_labels = all_files[:split_idx], all_labels[:split_idx]
    eval_files, eval_labels = all_files[split_idx:], all_labels[split_idx:]
    
    train_set = ASVSpoofDataset(train_files, train_labels)
    eval_set = ASVSpoofDataset(eval_files, eval_labels)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False, num_workers=2)
    
    model = ProsodicLCNNBiLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], device=device)
    class_weights = class_weights / class_weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_acc = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
        
        scheduler.step()
        
        train_acc = 100 * correct / len(train_set)
        
        model.eval()
        eval_correct = 0
        all_preds = []
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().tolist())
                eval_correct += (preds == y).sum().item()
        
        eval_acc = 100 * eval_correct / len(eval_set)
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"Eval Loss: {criterion(outputs, y).item():.4f} | Acc: {eval_acc:.2f}%")
        print(f"Eval Predictions: {Counter(all_preds)}")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")
    
    print(f"\nTraining complete! Best Eval Accuracy: {best_acc:.2f}%")
    return model

# Testing Function
def test_model(model_path, test_files, test_labels, device="cuda"):
    """Test the trained model on a given dataset."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ProsodicLCNNBiLSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_set = ASVSpoofDataset(test_files, test_labels)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)
    
    correct = 0
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            correct += (preds == y).sum().item()
    
    accuracy = 100 * correct / len(test_set)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Test Predictions: {Counter(all_preds)}")
    return accuracy

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Test with eval set (replace with new data if available)
    eval_dir = "/kaggle/input/asvspoof-2019-dataset/LA/ASVspoof2019_LA_eval/flac"
    test_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.flac')]
    test_labels = [1 if "LA_T_" in f else 0 for f in test_files]
    test_model("best_model.pth", test_files, test_labels)

# Previous Results (5 epochs):
# - Train Loss: 0.2496 | Acc: 88.53%
# - Eval Loss: 0.1184 | Acc: 89.50%
# - Eval Predictions: Counter({0: 13441, 1: 6022})
