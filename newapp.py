from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, torch, numpy as np
from PIL import Image
import re

# -------- Flask Setup --------
app = Flask(__name__)
CORS(app)  # allow frontend on localhost

# -------- YOLOv5 Load --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Loading YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to(DEVICE).eval()
print("[INFO] YOLOv5 Ready.")

# -------- BLIP Captioning (optional) --------
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to(DEVICE).eval()
    BLIP_AVAILABLE = True
except Exception as e:
    print("[WARN] BLIP not available:", e)
    BLIP_AVAILABLE = False

# -------- Helper Functions --------
def yolo_counts_from_pil(pil_image):
    np_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = yolo_model(np_img)
    df = results.pandas().xyxy[0]
    width, height = pil_image.size
    left_count = right_count = center_count = 0
    objects_with_dir = []
    for _, row in df.iterrows():
        name = str(row['name'])
        xmin = float(row['xmin']); xmax = float(row['xmax'])
        xcenter = (xmin + xmax) / 2.0
        if xcenter < width / 3: dirn = "left"; left_count += 1
        elif xcenter > 2 * width / 3: dirn = "right"; right_count += 1
        else: dirn = "center"; center_count += 1
        objects_with_dir.append(f"{name} on the {dirn}")
    return left_count, right_count, center_count, objects_with_dir

def blip_describe(pil_image):
    if not BLIP_AVAILABLE:
        return "Scene description not available."
    try:
        inputs = blip_processor(pil_image, return_tensors="pt").to(DEVICE)
        out_ids = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        return caption
    except:
        return "Could not describe the scene."

# -------- API Route --------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400
    file = request.files["image"]
    pil_image = Image.open(file.stream).convert("RGB")

    # YOLO object detection
    left_c, right_c, center_c, objs = yolo_counts_from_pil(pil_image)

    # Suggestion logic
    if center_c > 0:
        directions = "Stop. Obstacle ahead."
    elif left_c > right_c:
        directions = "Move right. Less obstacles on the right."
    elif right_c > left_c:
        directions = "Move left. Less obstacles on the left."
    else:
        directions = "Path is clear. Move forward."

    # BLIP captioning
    description = blip_describe(pil_image)

    return jsonify({
        "description": description,
        "directions": directions,
        "objects": objs
    })

# -------- Run --------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
