# Handwritten Character Recognition System

A high-performance web application designed to recognize handwritten characters, alphabets, and digits drawn interactively on a canvas or uploaded as images.

The application supports dual classification modes:
1. **Digits Mode**: Accurately classifies digits from `0` to `9`.
2. **Characters Mode**: Classifies alphabet characters (`A-Z`, `a-z`) dynamically.

---

## 🚀 Tech Stack

### Frontend
- **React (Vite)**: Clean, component-based user interface.
- **HTML5 Canvas API**: Interactive drawing canvas supporting mouse/touch inputs.
- **Vanilla CSS**: Premium dark-mode glassmorphic interface with micro-animations and custom fonts.

### Backend / ML Service
- **FastAPI (Python)**: REST API serving prediction requests.
- **Tesseract OCR (via Node.js Bridge)**: Built-in integration using `tesseract.js` (WebAssembly port of Tesseract 5.x) running portably inside Node.js. It requires no OS-level installers, no admin/UAC elevation, and features dynamic character whitelisting for high-confidence predictions.
- **Neural Network Training (Optional)**: Includes custom Python scripts for training CNN models on MNIST and EMNIST Balanced datasets using TensorFlow/Keras.

---

## 📁 Directory Structure

```
├── frontend/                 # Vite React App
│   ├── src/
│   │   ├── components/       # Drawing Canvas and Image Upload components
│   │   ├── App.jsx           # Main UI Assembly and state management
│   │   └── index.css         # Styling system
│   └── index.html
├── ml-service/               # FastAPI Backend Service
│   ├── models/               # Saved trained CNN models (.h5)
│   ├── ocr_bridge.js         # Node.js Tesseract.js worker bridge
│   ├── main.py               # Main FastAPI server application
│   └── test_predict.py       # API endpoint test script
├── training/                 # Offline CNN Training Pipeline
│   ├── download_mnist.py
│   ├── download_emnist.py
│   ├── train_digits.py       # MNIST training script
│   ├── train_characters.py   # EMNIST training script
│   └── train_all.py          # Sequential model training orchestrator
└── README.md
```

---

## 🛠️ Local Setup & Running

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **npm** (comes with Node.js)

### 2. Backend Setup
1. Navigate to the `ml-service` directory:
   ```bash
   cd ml-service
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies for Tesseract:
   ```bash
   npm install tesseract.js
   ```
4. Start the FastAPI backend server:
   ```bash
   python main.py
   ```
   The backend server will run at `http://localhost:8000`. You can verify it is healthy by visiting `http://localhost:8000/health`.

### 3. Frontend Setup
1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite React development server:
   ```bash
   npm run dev
   ```
   The application will run at `http://localhost:5173`. Open this URL in your web browser.

---

## 🧠 Optional: Training Custom CNN Models
If you wish to train custom convolutional neural network models from scratch:
1. Navigate to the `training` directory:
   ```bash
   cd training
   ```
2. Run the orchestrator script:
   ```bash
   python train_all.py
   ```
   This will automatically download the MNIST and EMNIST Balanced datasets, run the training scripts, output performance metrics, save confusion matrix plots under `training/plots/`, and save the trained models inside `ml-service/models/`.
