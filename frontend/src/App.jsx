import React, { useState, useEffect, useRef } from 'react';
import Canvas from './components/Canvas';
import ImageUpload from './components/ImageUpload';

function App() {
  const [mode, setMode] = useState('digits'); // 'digits' or 'characters'
  const [inputType, setInputType] = useState('draw'); // 'draw' or 'upload'
  const [brushSize, setBrushSize] = useState(14);
  const [uploadedImageBase64, setUploadedImageBase64] = useState(null);
  
  // API Health status
  const [apiStatus, setApiStatus] = useState('checking'); // 'checking', 'healthy', 'error'
  const [modelsLoaded, setModelsLoaded] = useState({ digits: false, characters: false });
  
  // Inference State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  
  const canvasRef = useRef(null);

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          const data = await response.json();
          setApiStatus('healthy');
          setModelsLoaded({
            digits: data.digits_model_loaded,
            characters: data.char_model_loaded
          });
        } else {
          setApiStatus('error');
        }
      } catch (err) {
        setApiStatus('error');
        setModelsLoaded({ digits: false, characters: false });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000); // Poll health every 5s
    return () => clearInterval(interval);
  }, []);

  const handleClear = () => {
    setResult(null);
    setErrorMsg(null);
    if (inputType === 'draw' && canvasRef.current && canvasRef.current.clear) {
      canvasRef.current.clear();
    }
  };

  const handlePredict = async () => {
    let imgBase64 = null;
    setErrorMsg(null);

    if (inputType === 'draw') {
      if (canvasRef.current) {
        // Run emptiness check
        const isEmpty = canvasRef.current.isEmpty ? canvasRef.current.isEmpty() : true;
        if (isEmpty) {
          setErrorMsg('Please draw something on the canvas first.');
          return;
        }
        if (canvasRef.current.getBase64Image) {
          imgBase64 = canvasRef.current.getBase64Image();
        }
      }
    } else {
      imgBase64 = uploadedImageBase64;
      if (!imgBase64) {
        setErrorMsg('Please upload an image first.');
        return;
      }
    }

    setLoading(true);
    try {
      console.log("Calling API URL:", "http://localhost:8000/predict");
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imgBase64,
          mode: mode
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to get prediction from backend');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || 'API connection failed. Make sure the backend server is running.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Determine if the current mode's model is loaded
  const currentModelLoaded = mode === 'digits' ? modelsLoaded.digits : modelsLoaded.characters;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-title-container">
          <h1 className="app-title">Handwritten Character Recognition</h1>
        </div>
        <p className="app-subtitle">Handwritten Digit & Character Recognition System using Convolutional Neural Networks</p>
      </header>

      {/* Main Grid */}
      <main className="main-grid">
        {/* Left Side: Input Pad */}
        <section className="glass-card">
          <h2 className="card-title">
            <span>✏️</span> Input Board
          </h2>

          {/* Mode Selector */}
          <div className="control-group" style={{ marginBottom: '20px' }}>
            <span className="control-label">RECOGNITION MODE</span>
            <div className="tab-container">
              <button 
                className={`tab-btn ${mode === 'digits' ? 'active' : ''}`}
                onClick={() => { setMode('digits'); handleClear(); }}
              >
                🔢 Digits (0-9)
              </button>
              <button 
                className={`tab-btn ${mode === 'characters' ? 'active' : ''}`}
                onClick={() => { setMode('characters'); handleClear(); }}
              >
                🔤 Characters (A-Z)
              </button>
            </div>
          </div>

          {/* Input Type Selector */}
          <div className="control-group" style={{ marginBottom: '24px' }}>
            <span className="control-label">INPUT METHOD</span>
            <div className="tab-container">
              <button 
                className={`tab-btn ${inputType === 'draw' ? 'active' : ''}`}
                onClick={() => { setInputType('draw'); handleClear(); }}
              >
                Canvas Draw
              </button>
              <button 
                className={`tab-btn ${inputType === 'upload' ? 'active' : ''}`}
                onClick={() => { setInputType('upload'); handleClear(); }}
              >
                Upload Image
              </button>
            </div>
          </div>

          {/* Canvas Wrapper or Image Uploader */}
          {inputType === 'draw' ? (
            <div className="canvas-wrapper">
              <Canvas 
                brushSize={brushSize} 
                onCanvasChange={() => { if (result) setResult(null); }}
                ref={canvasRef} 
              />
              
              {/* Brush size controller */}
              <div className="controls-panel">
                <div className="control-group">
                  <div className="control-label">
                    <span>Brush Width</span>
                    <span>{brushSize}px</span>
                  </div>
                  <input 
                    type="range" 
                    min="6" 
                    max="30" 
                    value={brushSize} 
                    onChange={(e) => setBrushSize(parseInt(e.target.value))}
                    className="brush-slider" 
                  />
                </div>
              </div>
            </div>
          ) : (
            <ImageUpload 
              onImageSelected={(base64) => {
                setUploadedImageBase64(base64);
                if (result) setResult(null);
              }} 
            />
          )}

          {/* Buttons */}
          <div className="btn-group" style={{ marginTop: '28px' }}>
            <button 
              className="btn btn-secondary" 
              onClick={handleClear}
              disabled={loading}
            >
              Clear
            </button>
            <button 
              className="btn btn-primary" 
              onClick={handlePredict}
              disabled={loading || (apiStatus === 'healthy' && !currentModelLoaded)}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Analyzing...
                </>
              ) : (
                'Classify Input'
              )}
            </button>
          </div>
        </section>

        {/* Right Side: Prediction & Results */}
        <section className="glass-card">
          <h2 className="card-title">
            <span>📊</span> Classifier Output
          </h2>

          <div className="results-wrapper">
            {errorMsg && (
              <div style={{ 
                background: 'rgba(239, 68, 68, 0.1)', 
                border: '1px solid rgba(239, 68, 68, 0.2)', 
                color: '#f87171',
                padding: '16px', 
                borderRadius: '12px',
                fontSize: '0.9rem',
                lineHeight: '1.4',
                animation: 'fadeIn 0.3s ease'
              }}>
                <strong>Error: </strong>{errorMsg}
              </div>
            )}

            {result ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
                {/* Highlighted winner */}
                <div className="winner-card">
                  <span className="winner-label">Detected Character</span>
                  <div className="winner-char">{result.prediction}</div>
                  <div className="winner-conf">
                    <span>Accuracy:</span>
                    <span>{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Confidences chart */}
                {result.top_predictions && (
                  <div className="chart-container">
                    <span className="chart-title">Confidence Distribution</span>
                    {result.top_predictions.map((item, idx) => (
                      <div key={idx} className="chart-row">
                        <span className="chart-row-label">{item.label}</span>
                        <div className="chart-bar-bg">
                          <div 
                            className="chart-bar-fill" 
                            style={{ width: `${item.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="chart-row-val">{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="prediction-placeholder">
                <div className="prediction-placeholder-icon">
                  {inputType === 'draw' ? '🖋️' : '🖼️'}
                </div>
                <div>
                  <p style={{ fontWeight: '600', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
                    Awaiting Character Input
                  </p>
                  <p style={{ fontSize: '0.85rem', maxWidth: '280px', margin: '0 auto' }}>
                    {inputType === 'draw' 
                      ? 'Draw a digit or letter on the canvas and click Classify Input.' 
                      : 'Upload an image file containing a hand-drawn character and click Classify Input.'
                    }
                  </p>
                </div>
              </div>
            )}

            {/* Extensibility Note */}
            <div className="extensibility-card">
              <div className="extensibility-title">
                <span>🚀</span> Future Sequence Modeling
              </div>
              <p>
                This system recognizes single characters. To recognize whole words/sentences, the pipeline can be extended with a <strong>CRNN (Convolutional Recurrent Neural Network)</strong>. In a CRNN, feature maps are extracted by a CNN, parsed sequentially by a bidirectional LSTM (to model context/dependencies), and decoded into full-text strings using <strong>CTC (Connectionist Temporal Classification) Loss</strong> without segmenting characters.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer Info & API Status */}
      <footer style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '16px' }}>
        <div className="api-badge">
          <div className={`api-dot ${apiStatus} ${apiStatus === 'healthy' && !currentModelLoaded ? 'error' : ''}`}></div>
          <span>
            API Status: {
              apiStatus === 'checking' ? 'Connecting to backend...' : 
              apiStatus === 'error' ? 'Offline (http://localhost:8000)' : 
              (currentModelLoaded ? 'Online & Model Loaded' : 'Online, but Model Not Trained')
            }
          </span>
        </div>

        {apiStatus === 'healthy' && (
          <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', textAlign: 'center' }}>
            Digits Model: {modelsLoaded.digits ? '✅ Loaded' : '❌ Not Loaded'} | 
            Characters Model: {modelsLoaded.characters ? '✅ Loaded' : '❌ Not Loaded'}
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;
