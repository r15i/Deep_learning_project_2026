# Deep Learning for Human Activity Recognition using WiFi Doppler Traces

This repository contains the official codebase and research paper evaluating various Deep Learning architectures on a Human Activity Recognition (HAR) system. The system leverages Doppler shifts extracted from commercial IEEE 802.11 (WiFi) Channel State Information (CSI) to identify both human activities and specific subjects in a non-invasive manner.

## Authors
- **Emilio Risi**   
- **Davide Pimazzoni** 

## Tasks
The project evaluates deep learning models on two primary tasks:
1. **Human Activity Recognition (HAR):** Classifying labeled physical activities including walking (W), running (R), jumping (J), sitting (L), and empty room (E). The models are trained on four subjects and strictly evaluated on two completely unseen subjects (Subjects 6 and 7).
2. **Contrastive Person Identification:** A metric learning task utilizing Triplet Margin Loss to identify and differentiate between specific subjects (Subjects 4 and 5). Testing is conducted on disjoint temporal sessions to evaluate environmental translation rather than simple zero-shot identity.

## Architectures Evaluated
1. **ResNet-8 (Micro-ResNet):** A specialized residual network tailored for our data dimensions. Proven to be highly effective, stable, and robust for Doppler spectrogram analysis.
2. **Multi-scale Inception Network:** Utilizes parallel convolutions with varying kernel sizes ($1\times1$, $2\times2$, $4\times4$) to capture spatial-temporal dynamics across different frequency scales simultaneously.
3. **Transformer-Encoder:** Explores the viability of multi-head self-attention on spectrograms. **Conclusion:** While convolutional models excel, the Transformer fails to generalize. Due to the extreme spatial sparsity of Doppler spectrograms and the Transformer's lack of a built-in inductive locality bias, the self-attention mechanism wastes capacity correlating empty space, leading to $100\%$ training memorization but complete structural overfitting on validation data.

## Data Pipeline
- **Dataset:** SHARP Doppler traces (Sanitized WiFi CSI).
- **Normalization:** Per-sample Instance Normalization (zero-mean, unit-variance) is applied dynamically during data loading to ensure robustness against varying signal transmission power across subjects.
- **Augmentation:** Employs Gaussian Noise ($\sigma=0.1$) and dynamic Time/Frequency Block Masking ($10\%$) to artificially induce robust representations.

## Repository Structure
- `architectures/`: PyTorch implementations of ResNet, Inception, and the Transformer.
- `core/`: Data loading pipelines (`dopplerdataset.py`), factory methods, and metric evaluation logic.
- `scripts/`: Utilities for deploying to Hyperstack, interacting with Weights & Biases (WandB) APIs, and plotting results (loss graphs, confusion matrices).
- `paper/`: LaTeX source files, assets, and the compiled PDF for the project report.
- `train.py`: Primary training pipeline.
- `Makefile`: Commands for syncing with remote GPU instances (Hyperstack) and compiling the paper.

## Usage

### Training
The training pipeline is entirely unified via `train.py` and supports dynamic architecture and task routing.

```bash
# Train ResNet-8 on Activity Recognition
python train.py --dataset-path /data/doppler_traces --arch resnet8 --task activity

# Train Inception on Contrastive Person ID
python train.py --dataset-path /data/doppler_traces --arch inception --task person_id

# Train Transformer on Activity Recognition
python train.py --dataset-path /data/doppler_traces --arch transformer --task activity
```

### Reproducing the Paper
The LaTeX source for our research paper is contained in the `paper/` directory. To compile the PDF report, simply run:
```bash
make compile-paper
```
This will generate `paper/template.pdf`.
