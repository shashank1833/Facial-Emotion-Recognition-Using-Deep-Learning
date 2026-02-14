# 🧠 Aura AI - Facial Emotion Recognition System

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node.js](https://img.shields.io/badge/node-20.19+-green.svg)
![React](https://img.shields.io/badge/react-19.2-blue.svg)

**A production-grade facial emotion recognition system powered by hybrid CNN-LSTM neural networks**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API](#-api)

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
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Aura AI is an advanced facial emotion recognition system that combines state-of-the-art deep learning techniques with a modern web interface. The system analyzes human emotions in real-time using a hybrid CNN-LSTM architecture that captures both spatial facial features and temporal emotion transitions.

### Key Capabilities

- **Real-time Analysis**: Process emotions from live webcam feeds with minimal latency
- **Multi-format Support**: Analyze static images (JPG/PNG) and video files (MP4/MOV)
- **High Accuracy**: Hybrid zone-based CNN architecture for precise emotion detection
- **Production Ready**: RESTful API with queue-based processing and error handling
- **Modern UI**: Beautiful, responsive React interface with 3D visual effects

---

## ✨ Features

### 🎭 **Emotion Recognition**
- Detects 7 emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, **Neutral**
- Real-time confidence scores and probability distributions
- Temporal analysis for video sequences

### 🖼️ **Input Modes**
- **Image Upload**: Drag-and-drop or file picker for static images
- **Video Analysis**: Process video files frame-by-frame
- **Live Detection**: Real-time webcam emotion streaming

### 🔬 **Advanced Processing**
- **Noise-Robust Preprocessing**: Median filtering and CLAHE normalization
- **Landmark Detection**: MediaPipe Face Mesh (468 facial landmarks)
- **Zone-Based Analysis**: Separate processing of eyes, mouth, forehead, and nose regions
- **Temporal Modeling**: LSTM layers capture emotion transitions

### 📊 **Analytics Dashboard**
- Historical analysis records with timestamps
- Emotion distribution statistics
- Performance metrics and data point tracking
- Export and data management features

---

## 🏗️ System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   React     │─────▶│   Node.js    │─────▶│    Python      │
│  Frontend   │      │   Backend    │      │   Inference    │
│  (Vite)     │◀─────│  (Express)   │◀─────│   Engine       │
└─────────────┘      └──────────────┘      └────────────────┘
     │                      │                       │
     │                      │                       │
  UI Layer            API Gateway            ML Processing
     │                      │                       │
     ├─ Image Upload        ├─ Request Queue        ├─ MediaPipe
     ├─ Webcam Capture      ├─ JSON Response        ├─ Zone CNNs
     ├─ Results Display     ├─ Error Handling       ├─ LSTM
     └─ Analytics           └─ Health Check         └─ Classifier
```

### Processing Pipeline

1. **Input** → User uploads image/video or activates webcam
2. **Frontend** → Captures frame and converts to base64
3. **Node.js Backend** → Queues request and spawns Python bridge
4. **Python Inference** → 
   - Face detection (MediaPipe)
   - Landmark extraction (468 points)
   - Zone segmentation (5 regions)
   - Feature extraction (Global + Zone CNNs)
   - Temporal analysis (LSTM)
   - Classification (7 emotions)
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

# If you get permission errors without virtual environment, use:
# pip install -r requirements.txt --break-system-packages
```

**Key Dependencies:**
- **PyTorch** (≥2.0.0): Deep learning framework
- **OpenCV** (≥4.8.0): Image processing
- **MediaPipe** (≥0.10.0): Face landmark detection
- **NumPy, Pandas**: Data processing
- **Albumentations, imgaug**: Data augmentation
- **TensorBoard**: Training visualization
- **PyYAML**: Configuration management

> **Note**: Using a virtual environment (recommended above) avoids the need for `--break-system-packages` flag.

#### Optional: GPU Support

For faster inference with CUDA-compatible GPUs:

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Optional: Alternative Face Detection

To use dlib instead of MediaPipe, uncomment the dlib line in `requirements.txt`:

```bash
# Uncomment in requirements.txt:
# dlib>=19.24.0

pip install dlib
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

# Download model (replace with your actual model URL)
# Place your trained model file as: checkpoints/best_model.pth
```

---

## ⚙️ Configuration

### Edit `configs/config.yaml`

```yaml
# Emotion labels
emotions:
  classes: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
  num_classes: 7

# Face detection
face_detection:
  method: "mediapipe"
  mediapipe:
    static_image_mode: false
    max_num_faces: 1
    min_detection_confidence: 0.5

# Model architecture
model:
  global_cnn:
    input_size: 224
    feature_dim: 512
  
  zone_cnn:
    input_size: 48
    feature_dim: 128

# Hardware
hardware:
  device: "cuda"  # Set to "cpu" if no GPU available
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

# Serve with backend
cd ../backend
node server.js
# Frontend will be served at http://localhost:5000
```

### Access the Application

Open your browser and navigate to:
- **Development**: `http://localhost:5173`
- **Production**: `http://localhost:5000`

### Using the Interface

#### 📸 **Image/Video Analysis**

1. Click on the **"Analyze"** tab
2. Drag and drop an image/video or click to upload
3. Click **"RUN INFERENCE"** button
4. View emotion prediction with confidence scores

#### 🎥 **Live Detection**

1. Click on the **"Live Detection"** tab
2. Click **"INITIALIZE CAMERA"**
3. Allow browser camera access
4. Click **"ANALYZE EMOTION"** to capture and analyze
5. Click **"TERMINATE SESSION"** to stop

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
│   ├── inference/
│   │   └── inference_utils.py        # Base inference class
│   ├── models/                       # CNN & LSTM architectures
│   ├── preprocessing/                # Image preprocessing
│   └── utils/                        # Helper functions
│
├── configs/
│   └── config.yaml                   # System configuration
│
├── checkpoints/                      # Trained model weights
│   └── best_model.pth
│
├── data/                             # Datasets (not included)
│   └── fer2013/
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
| Torchvision | ≥0.15.0 | Vision utilities |
| OpenCV | ≥4.8.0 | Image processing |
| MediaPipe | ≥0.10.0 | Face landmark detection |
| NumPy | ≥1.24.0 | Numerical computing |
| Pandas | ≥2.0.0 | Data manipulation |
| Scikit-learn | ≥1.3.0 | ML utilities |
| Albumentations | ≥1.3.0 | Data augmentation |
| TensorBoard | ≥2.13.0 | Training visualization |
| PyYAML | ≥6.0 | Configuration parsing |
| Matplotlib | ≥3.7.0 | Plotting |
| Seaborn | ≥0.12.0 | Statistical visualization |

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
# Reinstall PyTorch
pip install torch torchvision --break-system-packages
```

#### 3. **Model not found**

**Error:** `No model found`

**Solution:**
```bash
# Ensure model exists
ls -la checkpoints/best_model.pth

# Check path in inference_bridge.py
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

- **MediaPipe**: Face landmark detection
- **PyTorch**: Deep learning framework
- **FER-2013 Dataset**: Training data
- **React**: Frontend framework
- **Tailwind CSS**: UI styling

---

## 📧 Contact

For questions or support:

- **GitHub Issues**: [Create an issue](https://github.com/shashank1833/Facial-Emotion-Recognition-Using-Deep-Learning/issues)
- **Email**: shashankreddyremidi@gmail.com