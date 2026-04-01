# Facial Emotion Recognition System

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node.js](https://img.shields.io/badge/node-20.19+-green.svg)
![React](https://img.shields.io/badge/react-19.2-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Accuracy](https://img.shields.io/badge/accuracy-75.33%25-brightgreen.svg)

**A production-grade facial emotion recognition system powered by EfficientNet-B0, built with a three-tier React + Node.js + Python architecture.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Results](#-model-performance)

</div>

---

## 🎯 Overview

This is a full-stack facial emotion recognition application that classifies human facial expressions into 7 emotion categories in real time. The system uses an **EfficientNet-B0** convolutional neural network trained on a combined **AffectNet + RAF-DB** dataset and tested on **FER2013**, served through a Python inference engine, a Node.js/Express REST API, and a React/Vite frontend.

### Key Capabilities

- **Real-time Analysis** — Analyze emotions from live webcam feeds with low latency
- **Multi-format Input** — Supports static images (JPG/PNG) and webcam capture
- **High Accuracy** — 75.33% test accuracy across 7 emotion classes
- **Production Ready** — Queue-based REST API with error handling and health checks
- **Modern UI** — Responsive React interface with analytics dashboard

---

## ✨ Features

### 🎭 Emotion Recognition
- Detects 7 emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**
- Per-class confidence scores and full probability distribution
- Best performing class: **Happy** (F1: 0.888) · Lowest: **Disgust** (F1: 0.443)

### 🖼️ Input Modes
- **Image Upload** — Drag-and-drop or file picker for static images
- **Live Detection** — Real-time webcam emotion capture

### 📊 Analytics Dashboard
- Historical analysis records with timestamps
- Emotion distribution statistics
- Performance metrics and data export

---

## 🏗️ Architecture

The system follows a clean three-tier architecture:

```
┌──────────────────────┐
│   React Frontend     │  ← Vite dev server / production build
│   (Vite + Tailwind)  │    Image upload, webcam capture,
│                      │    results display, analytics dashboard
└──────────┬───────────┘
           │ HTTP / JSON (base64 image)
           ▼
┌──────────────────────┐
│  Backend Server      │  ← Node.js + Express
│  (Node.js/Express)   │    REST API gateway, request queue,
│                      │    error handling, health monitoring
└──────────┬───────────┘
           │ Spawns Python subprocess
           ▼
┌──────────────────────┐
│  ML Inference Engine │  ← Python service
│  (Python/PyTorch)    │    Image preprocessing,
│                      │    EfficientNet-B0 feature extraction,
│                      │    emotion classification
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ML Model            │
│  EfficientNet-B0     │  ← Pretrained backbone (ImageNet)
│  + FC Layer          │    Fine-tuned on RAF-DB (7 classes)
│  + Softmax           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output              │
│  - Emotion label     │
│  - Confidence score  │
│  - Probability dist. │
└──────────────────────┘
```

### Processing Pipeline

1. User uploads image or captures webcam frame via React frontend
2. Frontend encodes frame as base64 and sends POST request to Node.js backend
3. Node.js queues the request and spawns the Python inference bridge
4. Python service preprocesses the image and runs EfficientNet-B0 inference
5. Softmax output is returned as a JSON response with emotion label, confidence, and full probability distribution
6. Frontend displays results with visualizations and stores record in analytics

---

## 🔧 Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.8+ |
| Node.js | 20.19+ |
| npm | 8.0+ |
| Git | Latest |

**Hardware:**
- Minimum: 8 GB RAM, 2-core CPU
- Recommended: 16 GB RAM, 4-core CPU, CUDA-compatible GPU
- Webcam required for live detection

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shashank1833/Facial-Emotion-Recognition-Using-Deep-Learning.git
cd Facial-Emotion-Recognition-Using-Deep-Learning
```

### 2. Python Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt
```

**Key Python dependencies:**

| Package | Purpose |
|---------|---------|
| PyTorch ≥ 2.0.0 | Deep learning framework |
| Torchvision ≥ 0.15.0 | EfficientNet-B0 pretrained weights |
| OpenCV ≥ 4.8.0 | Image preprocessing |
| NumPy ≥ 1.24.0 | Numerical computing |
| Scikit-learn ≥ 1.3.0 | Evaluation metrics |
| Matplotlib / Seaborn | Training visualization |
| PyYAML ≥ 6.0 | Configuration parsing |

**Optional — GPU support (CUDA 11.8):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3. Backend Setup

```bash
cd backend
npm install
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

---

## 🧠 Model Training

The model uses **EfficientNet-B0** (pretrained on ImageNet) fine-tuned on a combined **AffectNet + RAF-DB** training set, evaluated on **FER2013**.

```bash
# Run training
python src/training/train.py \
  --config configs/config.yaml \
  --train_csv train.csv \
  --val_csv test.csv \
  --device cuda
```

**Training configuration (`configs/config.yaml`):**

```yaml
emotions:
  classes: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
  num_classes: 7

model:
  backbone: efficientnet_b0
  input_size: 224
  pretrained: true

hardware:
  device: "cuda"   # set to "cpu" if no GPU
```

---

## 🚀 Usage

### Start the Application

**Terminal 1 — Backend:**
```bash
cd backend
node server.js
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Access

| Mode | URL |
|------|-----|
| Development | `http://localhost:5173` |
| Production | `http://localhost:5000` |

### Using the Interface

**Image Analysis:**
1. Open the **Analyze** tab
2. Drag and drop an image or click to upload
3. Click **RUN INFERENCE**
4. View emotion prediction, confidence score, and probability distribution

**Live Detection:**
1. Open the **Live Detection** tab
2. Click **INITIALIZE CAMERA** and allow browser camera access
3. Click **ANALYZE EMOTION** to capture and classify
4. Click **TERMINATE SESSION** to stop

**Analytics:**
- **Records tab** — view analysis history with timestamps
- **Metrics tab** — see emotion distribution statistics

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Health Check
```http
GET /health
```
```json
{
  "status": "healthy",
  "backend": "nodejs",
  "python_bridge": "active"
}
```

#### Emotion Prediction
```http
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "image": "<base64_encoded_image>"
}
```

**Success (200):**
```json
{
  "emotion": "Happy",
  "confidence": 0.8542,
  "probabilities": {
    "Angry": 0.0234,
    "Disgust": 0.0123,
    "Fear": 0.0187,
    "Happy": 0.8542,
    "Sad": 0.0156,
    "Surprise": 0.0421,
    "Neutral": 0.0337
  },
  "detection_success": true
}
```

**Error (400):**
```json
{ "error": "No image data provided" }
```

**Error (500):**
```json
{ "error": "Inference timeout" }
```

**Notes:**
- Requests are processed sequentially via a queue
- Timeout: 30 seconds per request

---

## 📊 Model Performance

Trained for 15 epochs on **AffectNet + RAF-DB** (combined), tested on **FER2013**. Early stopping triggered at epoch 15; best checkpoint saved at epoch 5.

| Metric | Value |
|--------|-------|
| Overall Accuracy | 75.33% |
| Macro Precision | 0.6628 |
| Macro Recall | 0.6942 |
| Macro F1-Score | 0.6738 |
| Weighted F1-Score | 0.7590 |
| Final Train Accuracy | 95.95% |
| Best Val Loss | 2.3686 |

**Per-class breakdown:**

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.6412 | 0.6728 | 0.6566 |
| Disgust | 0.3905 | 0.5125 | 0.4432 |
| Fear | 0.5890 | 0.5811 | 0.5850 |
| Happy | 0.9465 | 0.8363 | **0.8880** |
| Sad | 0.7390 | 0.7050 | 0.7216 |
| Surprise | 0.6266 | 0.8723 | 0.7294 |
| Neutral | 0.7064 | 0.6794 | 0.6927 |

> **Note:** The large gap between training accuracy (95.95%) and validation accuracy reflects overfitting. Improvement areas: class-weighted loss for Disgust, stronger augmentation, and regularization.

---

## 📁 Project Structure

```text
emotion_recognition/
├── backend/                    # Node.js API Gateway
│   ├── server.js               # Express server with request queue
│   ├── inference_bridge.py     # Python bridge for model inference
│   ├── package.json            # Node.js dependencies
│   └── package-lock.json
│
├── frontend/                   # React Frontend (Vite + Tailwind)
│   ├── src/
│   │   ├── App.jsx             # Main application UI
│   │   ├── App.css             # UI styling
│   │   ├── index.css           # Global Tailwind styles
│   │   └── main.jsx            # React entry point
│   ├── public/                 # Static assets
│   ├── index.html              # HTML template
│   ├── vite.config.js          # Vite configuration
│   └── package.json            # Frontend dependencies
│
├── src/                        # Core Machine Learning Logic
│   ├── models/                 # Model Architecture
│   │   ├── hybrid_cnn.py       # EfficientNet-B0 backbone technique
│   │   ├── temporal_lstm.py    # Temporal modeling for video sequences
│   │   └── full_model.py       # Combined Hybrid CNN-LSTM model
│   ├── preprocessing/          # Image Preprocessing
│   │   └── noise_robust.py     # CLAHE & Median filtering
│   ├── landmark_detection/     # MediaPipe Face Mesh integration
│   │   └── mediapipe_detector.py
│   ├── zone_extraction/        # Facial Zone (Eyes/Mouth) logic
│   │   ├── zone_definitions.py
│   │   └── zone_extractor.py
│   ├── training/               # Training Pipeline
│   │   ├── data_loader.py      # Multi-dataset CSV loading
│   │   ├── augmentation.py     # Data augmentation strategies
│   │   ├── train.py            # Main training script
│   │   └── evaluate.py         # Model evaluation script
│   ├── inference/              # Inference Utilities
│   │   ├── inference_utils.py  # Model wrapper for predictions
│   │   ├── image_inference.py  # Static image testing
│   │   └── video_inference.py  # Real-time video processing
│   └── utils/                  # Helper Functions
│       ├── metrics.py          # F1, Accuracy, Confusion Matrix
│       └── visualization.py    # Performance graphing
│
├── configs/
│   └── config.yaml             # Hyperparameters & system configuration
│
├── checkpoints/                # Saved model weights (.pth)
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── ARCHITECTURE.md             # Detailed system design
├── LICENSE                     # MIT License
└── .gitignore                  # Git exclusion rules
```

---

## 🛠️ Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.2.0 | UI framework |
| Vite | 7.3.1 | Build tool & dev server |
| Tailwind CSS | 4.1.18 | Styling |
| Axios | 1.13.5 | HTTP client |
| Lucide React | 0.563.0 | Icons |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Node.js | 20.19+ | Runtime |
| Express | 5.2.1 | Web framework |
| CORS | 2.8.6 | Cross-origin requests |
| Body Parser | 2.2.2 | JSON parsing |

### Machine Learning
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | ML runtime |
| PyTorch | ≥ 2.0.0 | Deep learning |
| Torchvision | ≥ 0.15.0 | EfficientNet-B0 |
| OpenCV | ≥ 4.8.0 | Image processing |
| NumPy | ≥ 1.24.0 | Numerical computing |
| Scikit-learn | ≥ 1.3.0 | Metrics |
| Matplotlib / Seaborn | ≥ 3.7.0 | Visualization |
| PyYAML | ≥ 6.0 | Config parsing |

---

## 🐛 Troubleshooting

**Port already in use:**
```bash
lsof -i :5000
kill -9 <PID>
```

**Missing PyTorch module:**
```bash
pip install torch torchvision
```

**Model checkpoint not found:**
```bash
ls -la checkpoints/best_model.pth
# Ensure the path in inference_bridge.py matches
```

**Camera access denied:**
- Allow camera permissions in browser settings
- HTTPS required in production for camera access

**CORS errors:**
- Verify `API_BASE_URL` in `frontend/src/App.jsx` matches the backend URL
- Confirm CORS middleware is enabled in `backend/server.js`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

**Code style:** PEP 8 for Python · ESLint config for JavaScript · functional React components with hooks.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet) — Training dataset
- [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) — Training dataset
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — Test dataset
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Model backbone
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [React](https://react.dev/) — Frontend framework
- [Tailwind CSS](https://tailwindcss.com/) — UI styling

---

## 📧 Contact

- **GitHub Issues:** [Create an issue](https://github.com/shashank1833/Facial-Emotion-Recognition-Using-Deep-Learning/issues)
- **Email:** shashankreddyremidi@gmail.com