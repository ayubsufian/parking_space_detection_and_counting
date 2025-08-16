import cv2
import time
import pickle
import logging
import threading
import numpy as np
import re
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from keras.models import load_model
from waitress import serve

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the trained model once at startup
try:
    model = load_model('model/model_final.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

class_dictionary = {0: 'Empty', 1: 'Full'}

# Load parking space positions
with open('model/carposition.pkl', 'rb') as f:
    posList = pickle.load(f)

width, height = 130, 65

# Define default video capture sources
video_sources = {
    'webcam': 0,
    'ip_camera': 'rtsp://username:password@ip_address:554/stream',
    'video_file': 'assets/car_test.mp4'
}

class VideoCamera:
    def __init__(self, source):
        self.lock = threading.Lock()
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.running = True
        self.frame = None
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def update_frame(self):
        while self.running:
            with self.lock:
                if self.cap.isOpened():
                    success, frame = self.cap.read()
                    if success:
                        self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def switch_source(self, source):
        with self.lock:
            logging.info(f"Attempting to switch to {source}")
            self.cap.release()
            new_cap = cv2.VideoCapture(source)
            if new_cap.isOpened():
                self.cap = new_cap
                self.source = source
                logging.info(f"Switched to new video source: {source}")
            else:
                logging.error(f"Failed to open video source: {source}. Retaining old source.")
                # Reopen the old source to keep streaming
                self.cap = cv2.VideoCapture(self.source)

    def release(self):
        with self.lock:
            self.running = False
            self.cap.release()

# Initialize video camera with default source (video file)
camera = VideoCamera(video_sources['video_file'])

def check_parking_space(img):
    space_counter = 0
    img_crops = []

    for x, y in posList:
        crop = img[y:y + height, x:x + width]
        resized = cv2.resize(crop, (48, 48))
        normalized = resized / 255.0
        img_crops.append(normalized)

    img_crops = np.array(img_crops)
    predictions = model.predict(img_crops, batch_size=len(img_crops))

    for i, (x, y) in enumerate(posList):
        label_id = np.argmax(predictions[i])
        label = class_dictionary[label_id]

        if label == 'Empty':
            color = (100, 220, 100)
            text_color = (0, 0, 0)
            rectangle_color = (0, 255, 0)
            space_counter += 1
        else:
            color = (100, 100, 200)
            text_color = (255, 255, 255)
            rectangle_color = (0, 0, 255)

        thickness = 2
        font_scale = 0.5
        text_thickness = 1

        cv2.rectangle(img, (x, y), (x + width, y + height), rectangle_color, thickness)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        text_x, text_y = x, y + height - 5
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 6, text_y + 2), color, -1)
        cv2.putText(img, label, (text_x + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

    total_spaces = len(posList)
    return img, space_counter, total_spaces - space_counter

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        img = cv2.resize(frame, (1280, 720))
        img, free_spaces, occupied_spaces = check_parking_space(img)
        ret, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        time.sleep(1 / 30)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    frame = camera.get_frame()
    if frame is not None:
        img = cv2.resize(frame, (1280, 720))
        _, free_spaces, occupied_spaces = check_parking_space(img)
        return jsonify(free=free_spaces, occupied=occupied_spaces)
    return jsonify(free=0, occupied=0)

@app.route('/switch_source', methods=['POST'])
def switch_source():
    source_type = request.form.get('source')
    rtsp_url = request.form.get('rtsp-url')

    if rtsp_url:
        if validate_rtsp_url(rtsp_url):
            camera.switch_source(rtsp_url)
            return jsonify(status='success', source='user_rtsp', url=rtsp_url)
        else:
            return jsonify(status='error', message='Invalid RTSP URL format')

    if source_type in video_sources:
        camera.switch_source(video_sources[source_type])
        return jsonify(status='success', source=source_type)

    return jsonify(status='error', message='Invalid source or RTSP URL')

@app.route('/static/<path:filename>')
def static_files(filename):
    response = send_from_directory('static', filename)
    response.headers['Cache-Control'] = 'public, max-age=31536000'
    return response

def validate_rtsp_url(url):
    pattern = re.compile(
        r'^rtsp://(?:[a-zA-Z0-9]+(?::[a-zA-Z0-9]+)?@)?(?:[a-zA-Z0-9.-]+):?(\d+)?(?:/.*)?$'
    )
    return bool(pattern.match(url))

if __name__ == "__main__":
    try:
        serve(app, host='127.0.0.1', port=8000)
    finally:
        camera.release()
    ``