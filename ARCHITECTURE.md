# Technical Architecture: Hybrid CNN-LSTM

## 1. System Specifications

| Component | Input Dimension | Output Dimension | Description |
| :--- | :--- | :--- | :--- |
| **Global CNN** | 224x224x1 | 512 | Extracts high-level spatial context from the full face. |
| **Zone CNN (x5)** | 48x48x1 | 128 (each) | Extracts local micro-expression details (Eyes, Mouth, etc.). |
| **Feature Fusion** | 512 + (128x5) | 1152 | Concatenation of global and local features. |
| **Temporal LSTM** | Seq x 1152 | 7 | Processes temporal dynamics across 1-10 frames. |

---

## 2. Pipeline Data Flow

### Step A: Noise-Robust Preprocessing
1.  **Grayscale**: Image converted to single-channel (L).
2.  **Median Filter**: Kernel size 3x3 to remove salt-and-pepper noise.
3.  **CLAHE**: Clip limit 2.0, Tile grid size 8x8 for illumination balance.
4.  **Min-Max Norm**: Pixel values scaled to `[0.0, 1.0]`.

### Step B: Landmark-Based Extraction
- **Detector**: MediaPipe Face Mesh (468 points).
- **Zones**: 
    - `forehead`: Points around the brow line.
    - `left_eye` / `right_eye`: Orbital regions.
    - `nose`: Bridge and alar base.
    - `mouth`: Inner and outer lip contours.
- **Resolution**: All zones are normalized to **48x48 pixels** before entering the Zone CNNs.

---

## 3. Model Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam (Learning Rate: 1e-4) |
| **Loss Function** | Categorical Crossentropy with Class Weights |
| **Batch Size** | 32 |
| **Regularization** | Dropout (0.3), Batch Normalization |
| **Early Stopping** | Patience: 10, Min Delta: 0.001 |

---

## 4. Inference Logic
The system uses a **Sliding Window** approach for temporal analysis:
- For single images: Sequence length = 1 (padded).
- For video: Maintains a buffer of the last $N$ frames to capture expression transitions.
