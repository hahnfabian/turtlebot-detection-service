# TurtleBot Detection with a Custom YOLOv5 Model

This project provides a Flask API for detecting TurtleBot objects using a custom-trained YOLOv5 model.

The API accepts an image file via a POST request and returns detected TurtleBots along with their bounding boxes, class IDs, and confidence scores.

The main repository for the course project can be found here: [GitHub - TurtleBot Laser Tag](https://github.com/hahnfabian/turtlebot-lasertag).

## Installation

1. Clone the repository
2. Install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you have your own model, place it in the same directory as the script, or update the `model_path` variable to reflect the correct path.

## Usage 
1. Start the Flask server:
   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```
2. The API will be available at `http://localhost:5000`.
3. To detect TurtleBots, send a POST request to the `/detect` endpoint with an image file attached. Example usage:
   ```bash
   curl -X POST -F "file=@your_image.jpg" http://localhost:5000/detect
   ```

## Response
The API will return a JSON response with the following structure:
```json
{
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.95,
      "bounding_box": {
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.3
      }
    }
  ]
}
```
Or if there are no detections:
```json
{
  "message": "No objects detected." 
}
```


## Dataset
The was trained on a dataset of TurtleBot images. It is available here: [Dataset Link](https://huggingface.co/datasets/fhahn/turtlebot-detection-dataset-v1).
