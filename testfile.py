import os
import cv2
import dlib
import numpy as np
import onnx
from imutils import face_utils
import tensorflow as tf
import pickle
import onnxruntime as ort
from onnx_tf.backend import prepare

onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)

predictor = prepare(onnx_model)

print("hello world")