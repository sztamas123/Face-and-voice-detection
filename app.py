import argparse
import threading
import time

import cv2
import imutils
import numpy as np
from flask import Flask, render_template, Response, request
from imutils.video import FPS, VideoStream

from recognize_faces import detector, args, embedder, recognizer, le

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)


def gen_frames():
    # initialize the video stream
    global vs, outputFrame, lock
    fps = FPS().start()
    vs = VideoStream(src='/dev/video0').start()

    time.sleep(2.0)
    print("[INFO] starting video stream...")

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=2000)
        (h, w) = frame.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                # update the FPS counter
        fps.update()
        with lock:
            outputFrame = frame.copy()
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')

        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        #     break
    #fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    #cv2.destroyAllWindows()
    #vs.stop()


def generate():
    while outputFrame is True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
vs.stop()
