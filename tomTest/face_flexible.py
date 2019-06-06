import cv2
import grpc
import time
import numpy as np
import pymesh

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
route_table = simple_route_table

sess_id = "chain_face-000"
frame_id = 0

output_file = cv2.VideoWriter('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs/face_pipeline_output.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

while True:
  ret, image = cap.read()
  if (ret == 0):
    print("End of Video...")
    break

  frame_info = "%s-%s" % (sess_id, frame_id)
  route_index = 0

  request_input = dict()
  request_input['client_input'] = image
  request_input['frame_info'] = frame_info
  request_input['route_table'] = route_table
  request_input['route_index'] = route_index

  for i in range(len(route_table.split('-'))):
    current_model = route_table.split('-')[request_input['route_index']]

    if (current_model == "FaceDetector"):
      module_instance = FaceDetector()
    elif (current_model == "PRNetImageCropper"):
      module_instance = PRNetImageCropper()
    elif (current_model == "PRNet"):
      module_instance = PRNet()

    module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
    module_instance.Apply()
    next_request = module_instance.PostProcess(grpc_flag = False)

    next_request['frame_info'] = request_input['frame_info']
    next_request['route_table'] = request_input['route_table']
    next_request['route_index'] = request_input['route_index'] + 1

    request_input = next_request

    if (current_model == "PRNet"):
      # cv2.imwrite("tmp.jpg", request_input["FINAL"])
      output_file.write(request_input["FINAL"])

  frame_id += 1
