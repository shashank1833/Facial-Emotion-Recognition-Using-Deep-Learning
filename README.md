# 🧠 Aura AI - Facial Emotion Recognition System

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node.js](https://img.shields.io/badge/node-20.19+-green.svg)
![React](https://img.shields.io/badge/react-19.2-blue.svg)

**A facial emotion recognition system using transfer learning and deep neural networks**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API](#-api)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Aura AI is an image-based facial emotion recognition system that uses **transfer learning** with an EfficientNet-B0 backbone fine-tuned on facial expression datasets. It performs per-frame inference in real time.

### Key Capabilities

- **Real-time Analysis**: Process emotions from live webcam feeds with minimal latency
- **Multi-format Support**: Analyze static images (JPG/PNG) and video files (MP4/MOV)
- **RAF-DB**: Accuracy 78.75%; Macro F1 71.13%
- **Transfer Learning**: Leverages ImageNet pretrained models for robust feature extraction
- **REST API**: Sequential request handling with error management
- **UI**: Responsive React interface

---

## ✨ Features

### 🎭 **Emotion Recognition**
- Detects 7 emotions: **Surprise**, **Fear**, **Disgust**, **Happiness**, **Sadness**, **Anger**, **Neutral**
- Per-frame probabilities with confidence scores

### 🖼️ **Input Modes**
- **Image Upload**: Drag-and-drop or file picker for static images
- **Video Analysis**: Process video files frame-by-frame
- **Live Detection**: Real-time webcam, frame-by-frame inference

### 🔬 **Advanced Processing**
- **Preprocessing**: Resize to 224×224 and ImageNet normalization
- **Transfer Learning**: EfficientNet-B0 backbone pretrained on ImageNet
- **Data Augmentation**: Rotation, flip, color jitter, affine transforms for training robustness
- **Focal Loss**: Addresses class imbalance in emotion datasets

### 📊 **Analytics (Client-Side)**
- Historical analysis records with timestamps
- Emotion distribution statistics
- Performance metrics and data point tracking
- Export and data management features

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AURA AI ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   React     │─────▶│   Node.js    │─────▶│    Python      │
│  Frontend   │      │   Backend    │      │   Inference    │
│  (Vite)     │◀─────│  (Express)   │◀─────│   Engine       │
└─────────────┘      └──────────────┘      └────────────────┘
     │                      │                       │
     │                      │                       │
  UI Layer            API Gateway            ML Processing
     │                      │                       │
     ├─ Image Upload        ├─ Request Queue        ├─ Preprocessing
     ├─ Webcam Capture      ├─ JSON Response        ├─ EfficientNet-B0
     ├─ Results Display     ├─ Error Handling       ├─ Classifier
     └─ Analytics           └─ Health Check         └─ Softmax
```

### Processing Pipeline

1. **Input** → User uploads image/video or activates webcam
2. **Frontend** → Captures frame and converts to base64
3. **Node.js Backend** → Queues request and spawns Python bridge
4. **Python Inference** → 
   - Convert BGR → RGB
   - Resize to 224×224
   - Normalize (ImageNet mean/std)
   - Feature extraction (EfficientNet-B0 backbone)
   - Classification (custom head with dropout)
   - Softmax probabilities
5. **Response** → JSON with emotion, confidence, probabilities
6. **Frontend** → Displays results with visualizations

---

## 🔧 Prerequisites

### Required Software

- **Python**: 3.8 or higher
- **Node.js**: 20.19.0 or higher
- **npm**: 8.0.0 or higher
- **Git**: Latest version

### Hardware Requirements

- **Minimum**: 8GB RAM, 2-core CPU
- **Recommended**: 16GB RAM, 4-core CPU, GPU (CUDA compatible)
- **Webcam**: For live detection feature

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shashank1833/Facial-Emotion-Recognition-Using-Deep-Learning.git
cd Facial-Emotion-Recognition-Using-Deep-Learning
```

### 2️⃣ Backend Setup

#### Create Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

#### Install Python Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt
```

**Key Dependencies:**
- **PyTorch** (≥2.0.0): Deep learning framework
- **Torchvision** (≥0.15.0): Pretrained models and transforms
- **OpenCV** (≥4.8.0): Image processing
- **NumPy, Pandas**: Data processing
- **Albumentations**: Data augmentation
- **TensorBoard**: Training visualization
- **PyYAML**: Configuration management

> **Note**: Using a virtual environment is strongly recommended to avoid dependency conflicts.

#### Optional: GPU Support

For faster inference with CUDA-compatible GPUs:

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Install Node.js Dependencies

```bash
cd backend
npm install
```

**Dependencies installed:**
- `express`: Web framework
- `cors`: Cross-origin resource sharing
- `body-parser`: JSON request parsing

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
```

**Dependencies installed:**
- `react`, `react-dom`: UI framework
- `axios`: HTTP client
- `lucide-react`: Icon library
- `@tailwindcss/vite`: Styling framework
- `vite`: Build tool

### 4️⃣ Download Pre-trained Model

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download model (replace with your actual model URL or train your own)
# Place your trained model file as: checkpoints/best_model.pth
```

> **Note**: You need to train the model first or obtain a pretrained checkpoint. See [Training](#training-your-own-model) section.

---

## ⚙️ Configuration

### Edit `configs/config.yaml`

```yaml
# Emotion labels
emotions:
  classes: ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
  num_classes: 7

# Model architecture
model:
  backbone: "efficientnet_b0"  # Options: efficientnet_b0, mobilenet_v3_large
  pretrained: true
  input_size: 224
  dropout: 0.4

# Training
training:
  batch_size: 48
  epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.01
  optimizer: "adamw"
  loss: "focal_loss"
  focal_loss:
    alpha: 0.25
    gamma: 2.0

# Hardware
hardware:
  device: "cuda"  # Set to "cpu" if no GPU available
  num_workers: 4
  pin_memory: true
```

### Backend Configuration

Edit `backend/server.js` to change the port:

```javascript
const port = 5000;  // Default port
```

### Frontend Configuration

Edit `frontend/src/App.jsx` to update API URL:

```javascript
const API_BASE_URL = 'http://localhost:5000';
```

---

## 🚀 Usage

### Start the Application

#### Option 1: Separate Terminals (Development)

**Terminal 1 - Backend:**
```bash
cd backend
node server.js
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

#### Option 2: Production Build

```bash
# Build frontend
cd frontend
npm run build

# Preview production build (static server)
npm run preview
# Preview runs on http://localhost:4173 by default

# In a separate terminal, run backend API
cd ../backend
node server.js  # API at http://localhost:5000
```

### Access the Application

Open your browser and navigate to:
- **Development**: `http://localhost:5173`
- **Preview build**: `http://localhost:4173` (frontend), **Backend API**: `http://localhost:5000`

### Using the Interface

#### 📸 **Image/Video Analysis**

> Video and webcam inputs are processed frame-by-frame using the same image-based model; no temporal sequence modeling is performed.

1. Click on the **"Analyze"** tab
2. Drag and drop an image/video or click to upload
3. Click **"RUN INFERENCE"** button
4. View emotion prediction with confidence scores

#### 🎥 **Live Detection**

1. Click on the **"Live Detection"** tab
2. Click **"INITIALIZE CAMERA"**
3. Allow browser camera access
4. Click **"ANALYZE EMOTION"** to capture and analyze. Inference captures the current frame and processes it as a single image.
5. Click **"TERMINATE SESSION"** to stop
6. Camera runs only during Live Detection and stops when leaving the tab

#### 📊 **View Analytics**

- **Records Tab**: View analysis history
- **Metrics Tab**: See statistical breakdowns

---

## 📡 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "backend": "nodejs",
  "python_bridge": "active"
}
```

#### 2. Emotion Prediction

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "base64_encoded_image_data"
}
```

**Success Response (200):**
```json
{
  "emotion": "Happiness",
  "confidence": 0.8542,
  "probabilities": {
    "Surprise": 0.0234,
    "Fear": 0.0187,
    "Disgust": 0.0123,
    "Happiness": 0.8542,
    "Sadness": 0.0156,
    "Anger": 0.0421,
    "Neutral": 0.0337
  },
  "detection_success": true
}
```

**Error Response (400):**
```json
{
  "error": "No image data provided"
}
```

**Error Response (500):**
```json
{
  "error": "Inference timeout"
}
```

### Request Processing

- **Queue System**: Requests are processed sequentially
- **Timeout**: 30 seconds per request
- **Rate Limiting**: None (implement for production)

---

## 📊 Model Performance

### RAF-DB Test Set (In-Domain)

| Metric | Value |
|--------|-------|
| **Macro F1-Score** | 71.13% |
| **Accuracy** | 78.75% |
| **Macro Recall** | 72.83% |
| **Macro Precision** | 70.02% |

#### Per-Class Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Surprise | 75.07% | 86.02% | 80.17% |
| Fear | 61.19% | 55.41% | 58.16% |
| Disgust | 43.35% | 55.00% | 48.48% |
| **Happiness** | **95.91%** | **83.21%** | **89.11%** |
| Sadness | 75.15% | 77.82% | 76.46% |
| Anger | 65.95% | 75.31% | 70.32% |
| Neutral | 73.49% | 77.06% | 75.23% |

### FER-2013 Test Set (Cross-Domain)

| Metric | Value |
|--------|-------|
| **Accuracy** | 47.73% |
| **Macro F1-Score** | 40.05% |

> **Note**: Lower performance on FER-2013 is expected due to significant domain shift (grayscale, 48×48, in-the-wild conditions vs. RAF-DB's color, high-resolution, controlled images). These results reflect cross-dataset generalization, not training-time optimization.
> FER-2013 is used strictly for cross-dataset evaluation with no fine-tuning performed.

---

## 📁 Project Structure

```
emotion_recognition/
│
├── backend/                          # Node.js Backend
│   ├── server.js                     # Express server with queue processing
│   ├── inference_bridge.py           # Python inference wrapper
│   ├── package.json                  # Node.js dependencies
│   └── package-lock.json
│
├── frontend/                         # React Frontend
│   ├── src/
│   │   ├── App.jsx                   # Main application component
│   │   ├── App.css                   # Component styles
│   │   ├── index.css                 # Global styles with 3D effects
│   │   ├── main.jsx                  # React entry point
│   │   └── assets/                   # Images and icons
│   │
│   ├── public/                       # Static assets
│   ├── index.html                    # HTML template
│   ├── vite.config.js                # Vite configuration
│   ├── package.json                  # Frontend dependencies
│   └── eslint.config.js              # ESLint configuration
│
├── src/                              # Python ML Code
│   ├── models/
│   │   └── emotion_model.py          # EfficientNet/MobileNet model definition
│   │
│   ├── preprocessing/
│   │   └── noise_robust.py           # Preprocessing transforms (resize, normalization)
│   │
│   ├── training/
│   │   ├── train.py                  # Training script
│   │   ├── evaluate.py               # Evaluation script
│   │   ├── data_loader.py            # Dataset loaders (RAF-DB, FER-2013)
│   │   ├── losses.py                 # Focal Loss implementation
│   │   └── augmentation.py           # Data augmentation
│   │
│   ├── inference/
│   │   ├── inference_utils.py        # Shared inference utilities
│   │   ├── image_inference.py        # Single image inference
│   │   └── realtime_demo.py          # Webcam real-time demo
│   │
│   └── utils/
│       ├── metrics.py                # Evaluation metrics
│       └── visualization.py          # Plotting utilities
│
├── configs/
│   └── config.yaml                   # System configuration
│
├── checkpoints/                      # Trained model weights
│   └── best_model.pth
│
├── results/                          # Evaluation results
│   ├── metrics_raf_db.json
│   ├── metrics_fer2013.json
│   ├── confusion_matrix_raf_db.png
│   └── summary_raf_db.txt
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── ARCHITECTURE.md                   # Technical documentation
└── .gitignore                        # Git ignore rules
```

---

## 🛠️ Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.2.0 | UI framework |
| Vite | 7.3.1 | Build tool & dev server |
| Tailwind CSS | 4.1.18 | Styling framework |
| Axios | 1.13.5 | HTTP client |
| Lucide React | 0.563.0 | Icon library |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Node.js | 20.19+ | Runtime environment |
| Express | 5.2.1 | Web framework |
| CORS | 2.8.6 | Cross-origin requests |
| Body Parser | 2.2.2 | JSON parsing |

### Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | ML runtime |
| PyTorch | ≥2.0.0 | Deep learning framework |
| Torchvision | ≥0.15.0 | Pretrained models |
| EfficientNet | B0 | Backbone architecture |
| OpenCV | ≥4.8.0 | Image processing |
| NumPy | ≥1.24.0 | Numerical computing |
| Pandas | ≥2.0.0 | Data manipulation |
| Scikit-learn | ≥1.3.0 | Metrics |
| Albumentations | ≥1.3.0 | Data augmentation |
| TensorBoard | ≥2.13.0 | Training visualization |
| PyYAML | ≥6.0 | Configuration parsing |

---

## 🎓 Training Your Own Model

### Prepare Dataset

**RAF-DB Dataset Structure:**
```
data/raf-db/
├── DATASET/
│   ├── train/
│   │   ├── 1/  # Surprise
│   │   ├── 2/  # Fear
│   │   ├── 3/  # Disgust
│   │   ├── 4/  # Happiness
│   │   ├── 5/  # Sadness
│   │   ├── 6/  # Anger
│   │   └── 7/  # Neutral
│   └── test/
│       └── (same structure)
```

### Start Training

```bash
# Basic training
python src/training/train.py --config configs/config.yaml

# Monitor with TensorBoard
tensorboard --logdir logs/
```

### Evaluate Model

```bash
# Evaluate on RAF-DB and FER-2013
python src/training/evaluate.py \
  --model checkpoints/best_model.pth \
  --config configs/config.yaml \
  --results_dir results/
```

### Training Configuration

Key hyperparameters in `configs/config.yaml`:

```yaml
training:
  batch_size: 48
  epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.01
  loss: "focal_loss"  # Handles class imbalance
  
  early_stopping:
    enabled: true
    patience: 10
    
  augmentation:
    enabled: true
    rotation: 15
    horizontal_flip: true
    color_jitter:
      brightness: 0.2
      contrast: 0.2
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Backend won't start**

**Error:** `EADDRINUSE: address already in use`

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change port in server.js
```

#### 2. **Python bridge fails**

**Error:** `No module named 'torch'`

**Solution:**
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Reinstall PyTorch
pip install torch torchvision
```

#### 3. **Model not found**

**Error:** `No model found`

**Solution:**
```bash
# Ensure model exists
ls -la checkpoints/best_model.pth

# Train a new model or download pretrained
```

#### 4. **Camera access denied**

**Error:** `Camera access denied or not available`

**Solution:**
- Allow camera permissions in browser settings
- Use HTTPS for production (required for camera access)
- Check browser console for detailed errors

#### 5. **CORS errors**

**Error:** `Access-Control-Allow-Origin`

**Solution:**
- Verify CORS is enabled in `backend/server.js`
- Check API_BASE_URL in frontend matches backend URL

#### 6. **Low accuracy on custom images**

**Solution:**
- Ensure good lighting conditions
- Face should be clearly visible and frontal
- Image resolution should be reasonable (not too small)
- Model was trained on RAF-DB (relatively clean, frontal faces)

### Debug Mode

Enable verbose logging:

```bash
# Backend - already logs to stderr
node server.js

# Check browser console for frontend errors
# Press F12 → Console tab
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Use ESLint configuration provided
- **React**: Functional components with hooks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EfficientNet**: Google Research - Efficient deep learning architecture
- **RAF-DB**: Real-world Affective Faces Database
- **FER-2013**: Facial Expression Recognition 2013 dataset
- **PyTorch**: Deep learning framework
- **React**: Frontend framework
- **Tailwind CSS**: UI styling

---

## 📧 Contact

For questions or support:

- **GitHub Issues**: [Create an issue](https://github.com/shashank1833/Facial-Emotion-Recognition-Using-Deep-Learning/issues)
- **Email**: shashankreddyremidi@gmail.com

---

## 🔗 Quick Links

- [Model Architecture Details](ARCHITECTURE.md)
