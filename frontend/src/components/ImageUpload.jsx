import React, { useState, useRef } from 'react';

const ImageUpload = ({ onImageSelected }) => {
  const [dragActive, setDragActive] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;

    // Check if the file is an image
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const base64Data = e.target.result;
      setImagePreview(base64Data);
      onImageSelected(base64Data);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const clearImage = (e) => {
    e.stopPropagation();
    setImagePreview(null);
    onImageSelected(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div 
      className={`upload-zone ${dragActive ? 'drag-active' : ''} ${imagePreview ? 'has-preview' : ''}`}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      onClick={triggerFileInput}
      style={{
        border: '2px dashed rgba(255, 255, 255, 0.2)',
        borderRadius: '12px',
        padding: '24px',
        textAlign: 'center',
        background: dragActive ? 'rgba(59, 130, 246, 0.1)' : 'rgba(255, 255, 255, 0.03)',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        position: 'relative',
        minHeight: '232px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden'
      }}
    >
      <input 
        ref={fileInputRef}
        type="file" 
        className="file-input" 
        accept="image/*" 
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      
      {imagePreview ? (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img 
            src={imagePreview} 
            alt="Preview" 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '180px', 
              borderRadius: '8px',
              objectFit: 'contain',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
            }} 
          />
          <button 
            className="btn btn-secondary" 
            onClick={clearImage}
            style={{ 
              marginTop: '12px',
              padding: '6px 12px',
              fontSize: '0.85rem'
            }}
          >
            Remove Image
          </button>
        </div>
      ) : (
        <div className="upload-prompt" style={{ pointerEvents: 'none' }}>
          <div className="upload-icon" style={{ fontSize: '3rem', marginBottom: '8px', color: 'rgba(255, 255, 255, 0.4)' }}>
            📤
          </div>
          <p style={{ fontWeight: '500', marginBottom: '4px', fontSize: '1.05rem' }}>
            Drag & drop character image here
          </p>
          <p style={{ fontSize: '0.85rem', color: 'rgba(255, 255, 255, 0.4)' }}>
            or click to browse from files
          </p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
