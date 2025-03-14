# CNN-Based Denoiser Training

## Overview
This script trains a **CNN-based denoiser** using a dataset of noisy and clean spectrograms. The training process includes **logging**, **loss tracking**, **SNR calculation**, and **early stopping**. The trained model is saved for later inference.

## Features
- Uses **U-Net with ResNet101** backbone for denoising.
- Supports **modified MSE loss** for enhanced training.
- Logs detailed **batch-wise loss and SNR values**.
- Implements **early stopping** to prevent overfitting.
- Saves **training loss plots and spectrogram comparisons**.

## Prerequisites
### 1. Install Dependencies
Ensure you have Python and the required libraries installed:
```sh
pip install torch torchvision torchaudio numpy matplotlib
```

### 2. Check GPU Availability
The script automatically selects the **GPU** if available. To manually check:
```python
import torch
print(torch.cuda.is_available())
```
If `False`, ensure CUDA is properly installed.

## How to Run the Training
1. Navigate to the script directory:
```sh
cd /path/to/script
```

2. Run the training script:
```sh
python train_model.py
```

## Training Details
- **Data Loading**
  - The dataset is split into **train (70%)**, **validation (20%)**, and **test (10%)**.
  - The `SignalDataset` class handles spectrogram loading.
  
- **Logging**
  - A training log is created in `CNN_logs/`.
  - The model saves spectrograms in `CNN_spectrograms/`.
  
- **Loss Function**
  - Uses a **custom MSE loss function** (`modified_mse`).
  
- **SNR Computation**
  - Computes **noisy SNR** and **denoised SNR** at each batch.
  
- **Early Stopping**
  - Stops training if validation loss does not improve for **5 consecutive epochs**.

## Expected Output
- **Terminal Output:**
  ```sh
  Batch 10 - Train Loss: 0.023456 | Validation Loss: 0.019876
  Batch 10 - Noisy SNR = 5.21 dB | Denoised SNR = 12.45 dB
  ```

- **Logs:**
  The log file (`training_log_14_<timestamp>.py`) contains:
  ```
  Batch Number | Train Loss (AM/FM) | Train Loss (Chirp) | Train Loss (Trumpet) | Val Loss (AM/FM) | Val Loss (Chirp) | Val Loss (Trumpet) | Noisy SNR | Denoised SNR
  10 | 0.023 | 0.021 | 0.020 | 0.019 | 0.018 | 0.017 | 5.21 | 12.45
  ```

- **Saved Model:**
  - Model is saved as `CNN_Denoiser_model_14_<timestamp>.pth`.

- **Spectrogram Comparisons:**
  - Plots of **clean, noisy, and denoised spectrograms** are saved in `CNN_spectrograms/`.

## Customization
- **Modify Training Parameters** (batch size, learning rate, early stopping) in:
  ```python
  batch_size = 20
  learning_rate = 0.001
  early_stopping_patience = 5
  ```
- **Change Model Architecture**
  - By default, it uses `UNetResNet101()`. Uncomment `DenoisingAutoencoder()` to use a different model:
    ```python
    model = DenoisingAutoencoder().to(device)
    ```

## Troubleshooting
### Issue: `KeyError: 'signal_type' missing from batch`
**Solution:** Ensure `__getitem__` in `SignalDataset` correctly includes `signal_type`.

### Issue: `RuntimeError: CUDA out of memory`
**Solution:** Reduce `batch_size`:
```python
batch_size = 10
```

### Issue: Training loss stagnates
**Solution:** Try using a **lower learning rate**:
```python
learning_rate = 0.0005
```


