import os
import cv2
import grpc
import time
import numpy as np
import pickle

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
measure_module = "FaceDetector"
route_table = measure_module

sess_id = "chain_face-000"
frame_id = 0

# output_file = cv2.VideoWriter('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs/face_pipeline_output.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
output_file = cv2.VideoWriter('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs/face_pipeline_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

pickle_directory = "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % measure_module
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)


while (frame_id < 120):
  start = time.time()

  # get input
  if (measure_module == "FaceDetector"):
    ret, image = cap.read()
    frame_info = "%s-%s" % (sess_id, frame_id)
    route_index = 0
    request_input = dict()
    request_input['client_input'] = image
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  elif (measure_module == "PRNetImageCropper"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/FaceDetector", str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "PRNet"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/PRNetImageCropper", str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

  if (measure_module == "FaceDetector"):
    module_instance = FaceDetector()
  elif (measure_module == "PRNetImageCropper"):
    module_instance = PRNetImageCropper()
  elif (measure_module == "PRNet"):
    module_instance = PRNet()

  module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  next_request['frame_info'] = request_input['frame_info']
  next_request['route_table'] = request_input['route_table']
  next_request['route_index'] = request_input['route_index'] + 1

  end = time.time()
  print("duration = %s" % (end - start))

  # if (measure_module == "FaceDetector"):
  #   print(next_request["bounding_box"])
  # elif (measure_module == "PRNetImageCropper"):
  #   print(next_request["cropped_image"])
  #   print(next_request["tform_params"])
  # elif (measure_module == "PRNet"):
  #   output_file.write(next_request["FINAL"])

  # pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
  # with open(pickle_output, 'w') as f:
  #   pickle.dump(next_request, f)

  frame_id += 1

  # time.sleep(1.0)

# print(frame_id)