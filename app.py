import math
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.7)  # Adjusted for potentially faster detection

# Initialize the Classifier
classifier = Classifier("./lib/modelalpha-2/keras_model.h5", "./lib/modelalpha-2/labels.txt")

labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
          "v", "w", "x", "y"]

offset = 20
imgSize = 200  # Reduced size for faster processing
target_letter = 'a'  # Default target letter
predicted_letter = "None"


def process_frame(img, target_letter):
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    prediction = None
    global predicted_letter
    index = -1

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Adjust the aspect ratio calculation and resizing
        imgCrop = cv2.resize(img[y - offset:y + h + offset, x - offset:x + w + offset], (imgSize, imgSize))
        prediction, index = classifier.getPrediction(imgCrop, draw=False)
        predicted_letter = str(labels[index])
        logging.info(f"Prediction: {prediction}, Index: {index}")

        if str(labels[index]) == target_letter:
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)
        else:
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
    else:
        logging.info("No hand detected in the frame.")

    _, jpeg = cv2.imencode('.jpg', imgOutput, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # Increased compression
    return jpeg.tobytes(), prediction, index


@app.route('/set_target_letter', methods=['POST'])
def set_target_letter():
    global target_letter
    data = request.get_json()
    if 'letter' in data and len(data['letter']) == 1:
        target_letter = data['letter']
        return jsonify({'message': 'Target letter updated successfully'}), 200
    else:
        return jsonify({'message': 'Invalid input'}), 200

@app.route('/get_target_letter', methods=['GET'])
def get_predicted_letter():
    global predicted_letter
    return jsonify({'predicted_letter': predicted_letter}), 200

@socketio.on('image')
def handle_image(data):
    try:
        logging.info("Received frame for processing")

        # Simulate lower frame rate by sleeping (reduce server load)
        #time.sleep(0.1)  # Adjust this to decrease the effective frame rate

        sbuf = io.BytesIO()
        sbuf.write(base64.b64decode(data))
        pimg = Image.open(sbuf)
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

        # Process the frame and emit response
        processed_image, prediction, index = process_frame(frame,
                                                           target_letter)  # Assuming 'a' is the target letter for demonstration
        base64_image = base64.b64encode(processed_image).decode('utf-8')
        emit('image_response', {'image': base64_image})
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        emit('error', {'message': 'Error processing image'})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
