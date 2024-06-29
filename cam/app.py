from flask import Flask, render_template, Response
from webcam import process_frame
from model import load_model
import cv2
import os
import time

app = Flask(__name__, template_folder=os.path.abspath('templates'))
model = load_model('EfficientNet_best.ckpt')

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 프레임 처리 및 예측
            is_recyclable, predicted_frame = process_frame(frame, model)
            print(is_recyclable[0])
            print(predicted_frame)
            if is_recyclable == 1:
                text = "Recyclable"
            else:
                text = "Not Recyclable"

            # 프레임에 텍스트 추가
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)