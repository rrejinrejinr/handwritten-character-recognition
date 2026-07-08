import os
import base64
import re
import numpy as np
import subprocess
import json
from io import BytesIO
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Handwritten Character Recognition API")

# Enable CORS so the React frontend (running on http://localhost:5173) can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
digits_model = None
char_model = None

# Model files paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGITS_MODEL_PATH = os.path.join(BASE_DIR, "models", "digits_model.h5")
CHAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "char_model.h5")

# Class mappings
DIGIT_CLASSES = [str(i) for i in range(10)]
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

@app.on_event("startup")
def load_models():
    global digits_model, char_model
    import tensorflow as tf
    
    print("Loading models at startup...")
    if os.path.exists(DIGITS_MODEL_PATH):
        try:
            digits_model = tf.keras.models.load_model(DIGITS_MODEL_PATH)
            print("Successfully loaded digits model.")
        except Exception as e:
            print(f"Error loading digits model: {e}")
    else:
        print(f"Digits model not found at {DIGITS_MODEL_PATH}. Please train it first.")
        
    if os.path.exists(CHAR_MODEL_PATH):
        try:
            char_model = tf.keras.models.load_model(CHAR_MODEL_PATH)
            print("Successfully loaded characters model.")
        except Exception as e:
            print(f"Error loading characters model: {e}")
    else:
        print(f"Characters model not found at {CHAR_MODEL_PATH}. Please train it first.")

class PredictRequest(BaseModel):
    image: str  # Base64 encoded image
    mode: str   # "digits" or "characters"

def preprocess_image(base64_data: str) -> np.ndarray:
    """
    Decodes base64 image, converts to grayscale, auto-inverts to match MNIST/EMNIST format 
    (white stroke on black), crops to character bounding box, rescales preserving aspect ratio,
    centers it on a 28x28 canvas with padding, and normalizes pixels.
    """
    try:
        # Remove metadata header if present (e.g. data:image/png;base64,)
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
            
        img_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_data))
        
        # 1. Handle transparency / alpha channel
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Create a white background (standard for canvas drawing)
            background = Image.new("RGBA", img.size, (255, 255, 255))
            background.paste(img, (0, 0), img.convert("RGBA"))
            img = background.convert("L")
        else:
            img = img.convert("L")
            
        # 2. Auto-invert colors if the background is light
        # Convert to numpy to inspect border pixel intensities of the original drawing
        arr = np.array(img)
        border_pixels = np.concatenate([
            arr[0, :],      # Top edge
            arr[-1, :],     # Bottom edge
            arr[1:-1, 0],   # Left edge (excluding corners)
            arr[1:-1, -1]   # Right edge (excluding corners)
        ])
        
        # If average border is bright (white background), invert the image so it becomes white stroke on black
        if np.mean(border_pixels) > 127:
            img = ImageOps.invert(img)
            
        # 3. Add padding/centering logic: crop to character bounding box and center on 28x28 canvas
        bbox = img.getbbox()
        if bbox:
            cropped = img.crop(bbox)
            w, h = cropped.size
            max_dim = max(w, h)
            if max_dim == 0:
                max_dim = 1
            # EMNIST/MNIST characters are normalized to fit a 20x20 box inside a 28x28 frame
            scale = 20.0 / max_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            
            # Resize cropped image
            resized_char = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Center on a 28x28 black canvas
            centered_img = Image.new("L", (28, 28), color=0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            centered_img.paste(resized_char, (paste_x, paste_y))
            img = centered_img
        else:
            # Fallback to direct resize if canvas is empty
            img = img.resize((28, 28), Image.Resampling.BILINEAR)
            
        # 4. Save debug preprocessed image
        debug_path = os.path.join(BASE_DIR, "debug_preprocess.png")
        img.save(debug_path)
        print(f"Saved debug preprocess image to {debug_path}")
        
        # 5. Log and print the actual 28x28 array being sent to the model as ASCII Art
        final_arr = np.array(img)
        print("\n--- Preprocessed Image (28x28 ASCII Art) ---")
        for row in final_arr[::2, ::2]:  # Downsample to 14x14 for compact console print
            print("".join(["#" if val > 100 else " " for val in row]))
        print("--------------------------------------------\n")
        
        # 6. Convert to float32 (the model's internal Rescaling layer handles dividing by 255.0)
        float_arr = final_arr.astype('float32')
        
        # 7. Reshape to model expected input: (1, 28, 28, 1)
        final_input = np.expand_dims(float_arr, axis=(0, -1))
        
        return final_input
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def preprocess_for_tesseract(base64_data: str, dest_path: str):
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]
    img_data = base64.b64decode(base64_data)
    img = Image.open(BytesIO(img_data))
    
    # Handle transparency
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new("RGBA", img.size, (255, 255, 255))
        background.paste(img, (0, 0), img.convert("RGBA"))
        img = background.convert("L")
    else:
        img = img.convert("L")
        
    # Auto-invert to ensure black text on white background (which Tesseract expects)
    arr = np.array(img)
    border_pixels = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[1:-1, 0],
        arr[1:-1, -1]
    ])
    
    # If average border is dark (black background), invert it to white background
    if np.mean(border_pixels) < 127:
        img = ImageOps.invert(img)
        
    # Save image for Tesseract
    img.save(dest_path)

def run_tesseract_ocr(image_path: str, mode: str):
    try:
        # Run node bridge script
        result = subprocess.run(
            ["node", "ocr_bridge.js", image_path, mode],
            capture_output=True,
            text=True,
            check=True,
            cwd=BASE_DIR
        )
        # Parse JSON output
        output = json.loads(result.stdout.strip())
        return output.get("text", "?"), float(output.get("confidence", 0.0)) / 100.0
    except Exception as e:
        print(f"Error running Tesseract OCR: {str(e)}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Stderr: {e.stderr}")
        return "?", 0.0

@app.post("/predict")
def predict(request: PredictRequest):
    if request.mode not in ["digits", "characters"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'digits' or 'characters'")
        
    try:
        # 1. Define temp file path for the image
        temp_img_path = os.path.join(BASE_DIR, "temp_tesseract.png")
        
        # 2. Preprocess and save the image for Tesseract (black text on white background)
        preprocess_for_tesseract(request.image, temp_img_path)
        
        # 3. Run Tesseract OCR using our Node.js bridge
        predicted_label, confidence = run_tesseract_ocr(temp_img_path, request.mode)
        
        # 4. Clean up temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        # 5. Populate top_predictions with the single result
        top_predictions = [
            {
                "label": predicted_label,
                "confidence": confidence
            }
        ]
        
        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "digits_model_loaded": digits_model is not None,
        "char_model_loaded": char_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
