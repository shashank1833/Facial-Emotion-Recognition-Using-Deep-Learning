# Hybrid Facial Emotion Recognition (FER) System

A sophisticated, high-accuracy emotion recognition pipeline combining **Hybrid CNNs** for spatial feature extraction and **Temporal LSTMs** for sequence-based emotion classification.

---

## ⚡ Quick Start (Get Running in 3 Steps)

### 1. Environment Setup
```powershell
# Install core dependencies
pip install -r requirements.txt

# Setup Frontend & Backend
cd backend; npm install; cd ../frontend; npm install; cd ..
```

### 2. Data Preparation
Ensure your FER-2013 dataset is at `data/fer2013/fer2013.csv`.

### 3. Start Training or Demo
*   **To Train**: `python src/training/train.py --config configs/config.yaml --data data/fer2013/fer2013.csv`
*   **To Demo (Real-time)**: `python src/inference/realtime_demo.py`

---

## 🚀 Key Features
- **Hybrid CNN Architecture**: Combines a Global CNN (full face) with 5 dedicated Zone CNNs (Eyes, Nose, Mouth, Forehead).
- **Temporal Analysis**: Uses LSTM layers to process video sequences or multi-frame image inputs.
- **Robust Preprocessing**: Noise reduction (Median Filter), CLAHE for illumination normalization, and MediaPipe-based landmark detection.
- **Multi-Dataset Support**: Integrated support for FER-2013, CK+, and custom image folder datasets.
- **Full Stack Integration**: Includes a Python backend with a React frontend for real-time inference.

## 📂 Project Documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into the Hybrid CNN-LSTM technical specs and data flow.
- **[EMOTIONS.md](EMOTIONS.md)**: Guide to the 7 emotion classes and their visual detection cues.

## 📂 Project Structure
```text
emotion_recognition/
├── backend/                # Node.js/Python bridge for web-based inference
├── configs/                # Configuration files (hyperparameters, paths)
│   └── config.yaml         # Main configuration for training and model
├── data/                   # Dataset storage (FER-2013, etc.)
│   └── fer2013/            # Primary emotion dataset
├── frontend/               # React-based UI for real-time visualization
├── logs/                   # Training logs and TensorBoard events
├── src/                    # Source code
│   ├── inference/          # Image, video, and real-time inference scripts
│   ├── landmark_detection/ # MediaPipe-based face mesh logic
│   ├── models/             # Hybrid CNN and Temporal LSTM architectures
│   ├── preprocessing/      # Noise reduction and lighting normalization
│   ├── training/           # Data loaders and training loops
│   ├── utils/              # Metrics and visualization helpers
│   └── zone_extraction/    # Dynamic facial zone cropping logic
├── test_batch_images.py    # Batch testing utility
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## 🛠️ Advanced Installation
1. Clone the repository.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install frontend/backend dependencies:
   ```bash
   cd frontend && npm install
   cd ../backend && npm install
   ```

## 📈 Detailed Training
To start a new training run with the optimized 7-emotion configuration:
```bash
python src/training/train.py --config configs/config.yaml --data data/fer2013/fer2013.csv
```

## 🧪 Testing & Validation
Run batch testing on the FER-2013 test set to verify per-emotion accuracy:
```bash
python test_batch_images.py
```

## 📜 License
This project is licensed under the MIT License.
