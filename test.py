import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from modules_avatar.prnet import PRNet
from modules_avatar.prnet_image_cropper import PRNetImageCropper
from modules_avatar.face_detector import FaceDetector

import tensorflow as tf

# For plotting
from PRNet.utils.cv_plot import plot_kpt, plot_vertices

import time
import numpy as np
# import pymesh
import threading
from Queue import Queue


q = Queue()
def worker():
   while True:
       item = q.get()
       image = np.zeros((frame_height, frame_width))
       if item is not None:
           vertices = item
           show_img = plot_vertices(np.zeros_like(image), vertices)
       else:
           show_img = image
               # Display the resulting frame
       cv2.imshow('frame',show_img)

       # Press Q on keyboard to stop recording
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
       q.task_done()

t = threading.Thread(target=worker)
t.daemon = True
t.start()



channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Setup
PRNet.Setup()
FaceDetector.Setup()
PRNetImageCropper.Setup()


cap = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/IMG_0003.mov")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = None
depth_out = None
frame_num = 1490
ind = 0
while frame_num:
   frame_num -= 1
   ret, image = cap.read()
   if ret == 0:
       break
   if out is None:
       out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
           'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
       # depth_out = cv2.VideoWriter('output_depth.avi', cv2.VideoWriter_fourcc(
       #     'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

   pp_start_time = time.time()
   start_time = time.time()
   next_request = predict_pb2.PredictRequest()
   next_request.inputs['input_image'].CopyFrom(
     tf.make_tensor_proto(image))
   elapsed_time = time.time() - start_time
   print('serialization time cost: {}'.format(elapsed_time))

   start_time = time.time()
   face_detector = FaceDetector()
   face_detector.PreProcess(next_request, stub)
   face_detector.Apply()
