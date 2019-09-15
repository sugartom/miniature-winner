import os
import cv2
import grpc
import time
import numpy as np
import pickle
import sys
import threading

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

from modules_avatar.face_detector_flexible import FaceDetector
from modules_avatar.prnet_image_cropper_flexible import PRNetImageCropper
from modules_avatar.prnet_flexible import PRNet

# from PRNet.utils.cv_plot import plot_vertices

ichannel = grpc.insecure_channel('0.0.0.0:8500')
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# Setup
FaceDetector.Setup()
PRNetImageCropper.Setup()
PRNet.Setup()

cap = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/IMG_0003.mov")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

simple_route_table = "FaceDetector-PRNetImageCropper-PRNet"
measure_module = "PRNet"
route_table = measure_module

sess_id = "chain_face-000"
frame_id = 0

input_fps = int(sys.argv[1])
total_frame = 120

def runFrame(measure_module, request_input, frame_id):
  if (measure_module == "FaceDetector"):
    module_instance = FaceDetector()
  elif (measure_module == "PRNetImageCropper"):
    module_instance = PRNetImageCropper()
  elif (measure_module == "PRNet"):
    module_instance = PRNet()

  module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  print("Finished frame %d for module %s" % (frame_id, measure_module))

  if (frame_id == 10):
    global stime
    stime = time.time()
  elif (frame_id == total_frame - 1):
    global etime
    etime = time.time()

if True:
# get input
  if (measure_module == "FaceDetector"):
    ret, image = cap.read()
    frame_info = "%s-%s" % (sess_id, "32")
    route_index = 0
    request_input = dict()
    request_input['client_input'] = image
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  elif (measure_module == "PRNetImageCropper"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/FaceDetector", str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "PRNet"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/PRNetImageCropper", str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

while frame_id < total_frame:
  # frame_thread = threading.Thread(target = runFrame, args = (measure_module, sess_id, frame_id, reader,))
  frame_thread = threading.Thread(target = runFrame, args = (measure_module, request_input, frame_id,))
  frame_thread.start()

  time.sleep(1.0/input_fps)
  frame_id += 1

try:
  while True:
    time.sleep(60 * 60 * 24)
except KeyboardInterrupt:
  print("\nEnd by keyboard interrupt")
  print("<%f, %f> = %f over %d frames with fps of %f" % (float(stime), float(etime), float(etime) - float(stime), total_frame, (total_frame - 1 - 10) / (float(etime) - float(stime))))
