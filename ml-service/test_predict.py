import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw

def test_api():
    print("=== Testing FastAPI Handwritten Recognition API ===")
    
    url = "http://localhost:8000/predict"
    
    # Function to query and print prediction
    def predict_char(img, mode_name, char_name):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = {
            "image": f"data:image/png;base64,{img_str}",
            "mode": mode_name
        }
        
        print(f"\n--- Testing '{char_name}' in '{mode_name}' mode ---")
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction:", result["prediction"])
            print("Confidence:", f"{result['confidence'] * 100:.2f}%")
            print("Top 3 Predictions:")
            for idx, pred in enumerate(result["top_predictions"][:3]):
                print(f"  {idx+1}. Class: {pred['label']}, Confidence: {pred['confidence'] * 100:.2f}%")
        else:
            print("Error:", response.text)

    # 1. Create a digit '3' (curved)
    img_3 = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw_3 = ImageDraw.Draw(img_3)
    # Draw a curved digit 3
    draw_3.arc([30, 20, 70, 50], start=180, end=0, fill=(0, 0, 0), width=8)
    draw_3.line([(70, 35), (45, 50)], fill=(0, 0, 0), width=8)
    draw_3.arc([30, 50, 70, 80], start=270, end=180, fill=(0, 0, 0), width=8)
    predict_char(img_3, "digits", "3 (Curved)")

    # 2. Create a letter 'A'
    img_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw_a = ImageDraw.Draw(img_a)
    # Left diagonal
    draw_a.line([(50, 15), (25, 80)], fill=(0, 0, 0), width=8)
    # Right diagonal
    draw_a.line([(50, 15), (75, 80)], fill=(0, 0, 0), width=8)
    # Cross bar
    draw_a.line([(35, 55), (65, 55)], fill=(0, 0, 0), width=8)
    predict_char(img_a, "characters", "A")

    # 3. Create a letter 'Z'
    img_z = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw_z = ImageDraw.Draw(img_z)
    # Top horizontal
    draw_z.line([(25, 25), (75, 25)], fill=(0, 0, 0), width=8)
    # Diagonal down-left
    draw_z.line([(75, 25), (25, 75)], fill=(0, 0, 0), width=8)
    # Bottom horizontal
    draw_z.line([(25, 75), (75, 75)], fill=(0, 0, 0), width=8)
    predict_char(img_z, "characters", "Z")

if __name__ == '__main__':
    test_api()
