from flask import Flask, render_template, Response
import cv2
from imutils.video import VideoStream

import recognize_faces

app = Flask(__name__)
#cv2.VideoCapture(-1)
#camera = VideoStream(src=0).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug = True)
