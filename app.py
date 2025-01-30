from flask import Flask, request, jsonify
import torch
import cv2
import json
import numpy as np

app = Flask(__name__)

model_path = 'best.pt'  # Path to your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR) 

    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    results = model(img)
    detections = results.xywh[0] 

    output = {
        "detections": []
    }

    if len(detections) > 0:
        for detection in detections:
            class_id = int(detection[5])  
            confidence = float(detection[4]) 
            x_center, y_center, width, height = map(float, detection[:4])  

            output["detections"].append({
                "class_id": class_id,
                "confidence": confidence,
                "bounding_box": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }
            })

    if not output["detections"]:
        output["message"] = "No objects detected."

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
