import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

import tensorflow as tf

import time
import numpy as np

# import threading
# from Queue import Queue

# # Set up visualizer channel
# visualize_channel = grpc.insecure_channel("localhost:50051")
# visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)

worker_channel = grpc.insecure_channel("192.168.1.9:50101")
worker_stub = prediction_service_pb2_grpc.PredictionServiceStub(worker_channel)

# # Setup a worker to send visualization
# q = Queue()
# def worker():
#   while True:
#     print("test")
#     item = q.get()
#     result = visualize_stub.Predict(item, 10.0)

#     q.task_done()

# t = threading.Thread(target = worker)
# t.daemon = True
# t.start()

cap = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/IMG_0003_720.mp4")
# cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_id = 0
read_time = -1

while True:
  ret, image = cap.read()
  if (ret == 0):
    break

  if (time.time() - read_time < 1.0/15):
    continue
  read_time = time.time()

  frame_id += 1

  face_request = predict_pb2.PredictRequest()
  face_request.inputs['input_image'].CopyFrom(
    tf.make_tensor_proto(image))

  face_result = worker_stub.Predict(face_request, 10.0)

  # if ("prnet_output" in face_result.outputs):
  #   key_points = tensor_util.MakeNdarray(face_result.outputs["prnet_output"])
  #   vertices = tensor_util.MakeNdarray(face_result.outputs["vertices"])

  #   visualizer_request = predict_pb2.PredictRequest()
  #   visualizer_request.inputs['prnet_output'].CopyFrom(
  #     tf.make_tensor_proto(key_points))
  #   visualizer_request.inputs['vertices'].CopyFrom(
  #     tf.make_tensor_proto(vertices))

  #   q.put(visualizer_request)
