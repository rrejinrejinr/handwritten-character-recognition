import React, { useRef, useState, useEffect, useImperativeHandle, forwardRef } from 'react';

const Canvas = forwardRef(({ brushSize, onCanvasChange }, ref) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastCoords, setLastCoords] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      // Set canvas drawing resolution separate from CSS size
      canvas.width = 280;
      canvas.height = 280;
      
      const ctx = canvas.getContext('2d');
      // Initialize with solid black background
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Set default stroke styles
      ctx.strokeStyle = '#ffffff';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    }
  }, []);

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    const rect = canvas.getBoundingClientRect();
    
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
      // Prevent scrolling on touch devices when drawing
      if (e.cancelable) e.preventDefault();
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    
    // Account for styling vs actual resolution
    const x = ((clientX - rect.left) / rect.width) * canvas.width;
    const y = ((clientY - rect.top) / rect.height) * canvas.height;
    
    return { x, y };
  };

  const startDrawing = (e) => {
    const coords = getCoordinates(e);
    if (!coords) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = '#ffffff';
    
    // Draw a single dot on click/touch down
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, brushSize / 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    
    setIsDrawing(true);
    setLastCoords(coords);
    
    if (onCanvasChange) {
      onCanvasChange();
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;
    
    const coords = getCoordinates(e);
    if (!coords) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.lineWidth = brushSize;
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    setLastCoords(coords);
    
    if (onCanvasChange) {
      onCanvasChange();
    }
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
  };

  // Exposed method to clear the canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (onCanvasChange) {
        onCanvasChange();
      }
    }
  };

  // Expose control functions to parent using useImperativeHandle
  useImperativeHandle(ref, () => ({
    clear: clearCanvas,
    getBase64Image: () => {
      if (canvasRef.current) {
        return canvasRef.current.toDataURL('image/png');
      }
      return null;
    },
    isEmpty: () => {
      if (!canvasRef.current) return true;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      
      // Count non-black pixels
      let nonZeroCount = 0;
      for (let i = 0; i < data.length; i += 4) {
        // Red, Green, or Blue channels are non-zero (since background is solid black #000000)
        if (data[i] > 0 || data[i+1] > 0 || data[i+2] > 0) {
          nonZeroCount++;
        }
      }
      
      console.log(`[Canvas Emptiness Check] Total non-zero pixels: ${nonZeroCount}`);
      // Consider empty if there are fewer than 10 non-zero pixels (allow minor noise)
      return nonZeroCount < 10;
    }
  }));

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        style={{
          touchAction: 'none', // Prevents default gesture behaviors (like scrolling)
          cursor: 'crosshair',
          width: '100%',
          height: '100%',
          borderRadius: '8px',
          display: 'block',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
        }}
      />
    </div>
  );
});

export default Canvas;
