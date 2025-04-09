# Documentation & Analysis

## Implementation Process
- **Challenges**: 
  - Initial overfitting (100% train, 0% eval) due to dataset imbalance.
  - Dimension mismatch in CNN and prosodic features (50 vs. 99 time steps).
- **Solutions**: 
  - Merged and resplit ASVspoof 2019 train/eval sets for balance.
  - Adjusted prosodic interpolation to 50 time steps.
- **Assumptions**: 
  - `LA_T_` files are spoofed, others bonafide (simplified labeling).
  - 5 epochs suffice for initial results (later extended to 10).

## Model Selection
- **Why Chosen**: The Prosodic + Pronunciation (LCNN + Bi-LSTM) approach leverages speech-specific features, aligning with real conversation analysis. My prior 89.50% accuracy with a similar setup confirmed its potential.
- **How It Works**: 
  - LCNN processes Mel-spectrograms for spectral features.
  - Prosodic extractor adds pitch/energy cues.
  - Bi-LSTM models temporal patterns, followed by a classifier.

## Performance Results
- **Dataset**: ASVspoof 2019 LA (77,850 train, 19,463 eval samples).
- **Results**: Best eval accuracy 89.50% (Epoch 5), train loss 0.2496, eval loss 0.1184.
- **Strengths**: Good generalization, balanced predictions (13,441 bonafide, 6,022 spoof).
- **Weaknesses**: Slight overprediction of spoofs; eval dip at Epoch 2 suggests instability.

## Future Improvements
- Train for 10+ epochs to push accuracy past 90%.
- Add EER metric for spoof detection benchmarking.
- Incorporate phoneme features (e.g., via Conformer) as in the original approach.

## Reflection Questions
1. **Significant Challenges**: Dataset imbalance and feature alignment were toughest; resolved via resplitting and debugging dimensions.
2. **Real-World vs. Research**: May struggle with noisy real-world audio; needs robustness testing.
3. **Additional Resources**: More diverse datasets (e.g., ASVspoof 5) and pre-trained phoneme models could boost performance.
4. **Deployment**: Optimize for speed (reduce CNN layers), use ONNX for inference, and deploy with a streaming audio pipeline.
