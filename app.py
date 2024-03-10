import math

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time
import logging

logging.basicConfig(level=logging.INFO)


app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Initialize the Classifier
classifier = Classifier("./lib/modelalpha-2/keras_model.h5", "./lib/modelalpha-2/labels.txt")

labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
          "v", "w", "x", "y"]

offset = 20
imgSize = 300

def process_frame(img, target_letter):
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    prediction = None
    index = -1

    if hands:
        logging.info(f"Hand detected. Number of hands: {len(hands)}")
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        logging.info(f"Prediction: {prediction}, Index: {index}")

        prediction = prediction.tolist() if prediction is not None else None
        index = int(index) if index is not None else None

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

    _, jpeg = cv2.imencode('.jpg', imgOutput)
    return jpeg.tobytes(), prediction, index

@socketio.on('connect')
def test_connect():
    print("Client connected")

@socketio.on('disconnect')
def test_disconnect():
    print("Client disconnected")

@socketio.on('image')
def handle_image(data):
    # Log when a new frame is received
    logging.info("Received frame for processing")

    # Decode the image from base64
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(data))
    pimg = Image.open(sbuf)

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Log before starting the processing
    logging.info("Starting frame processing")

    # Process the frame
    imgOutput = frame.copy()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if str(labels[index]) == "a":
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

    # Log after processing the frame
        logging.info(f"Finished processing frame with prediction: {prediction}, index: {index}")

    success, encoded_image = cv2.imencode('.jpg', imgOutput)
    if not success:
        logging.error('Could not encode image to JPEG format')
        return

    # Step 2: Convert the encoded image to a base64 string
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    # Step 3: Emit the base64 string to the client
    emit('image_response', {'image': base64_image})
    # Convert processed image back to base64 for transmission



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
