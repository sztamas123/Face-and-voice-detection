from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import subprocess

import os
import sys
import time


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            pass #quit

        self.status = QStatusBar()
        self.setStatusBar(self.status)


        self.save_path = ""

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.setCentralWidget(self.viewfinder)

        # Set the default camera.
        self.select_camera(0)

        # Setup tools
        camera_toolbar = QToolBar("Camera")
        camera_toolbar.setIconSize(QSize(60, 60))
        self.addToolBar(camera_toolbar)

        photo_action = QAction(QIcon(os.path.join('icons', 'camera.png')), "Take photo...", self)
        photo_action.setStatusTip("Take photo of current view")
        photo_action.triggered.connect(self.take_photo)
        camera_toolbar.addAction(photo_action)

        change_folder_action = QAction(QIcon(os.path.join('icons', 'folder.png')), "Change save location...", self)
        change_folder_action.setStatusTip("Change folder where photos are saved.")
        change_folder_action.triggered.connect(self.change_folder)
        camera_toolbar.addAction(change_folder_action)


        camera_selector = QComboBox()
        camera_selector.addItems([c.description() for c in self.available_cameras])
        camera_selector.currentIndexChanged.connect( self.select_camera )


        extract_images = QAction(QIcon(os.path.join('icons', 'extract.png')), "Extract", self)
        extract_images.setStatusTip("Extract embeddings of the photos.")
        extract_images.triggered.connect(self.train)
        camera_toolbar.addAction(extract_images)

        run_app = QAction(QIcon(os.path.join('icons', 'extract.png')), "Run", self)
        run_app.setStatusTip("Run the face recognizer")
        run_app.triggered.connect(self.startApp)
        camera_toolbar.addAction(run_app)



        camera_toolbar.addWidget(camera_selector)


        self.setWindowTitle("Add picture")
        self.show()

    def select_camera(self, i):
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()

        self.capture = QCameraImageCapture(self.camera)
        self.capture.error.connect(lambda i, e, s: self.alert(s))
        self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image %04d captured" % self.save_seq))

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0

    def take_photo(self):
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        self.capture.capture(os.path.join(self.save_path, "%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        )))
        self.save_seq += 1

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0


    def train(self):
        subprocess.call(["./extract.sh"])

    def startApp(self):
        subprocess.call(["face_rec.sh"])

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        err = QErrorMessage(self)
        err.showMessage(s)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("Add picture")

    window = MainWindow()
    window.setGeometry(0,0,3000,3000)
    app.exec_()