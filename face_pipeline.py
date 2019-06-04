import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from modules.prnet import PRNet
from modules.prnet_image_cropper import PRNetImageCropper
from modules.face_detector import FaceDetector

import tensorflow as tf

# For plotting
from PRNet.utils.cv_plot import plot_kpt, plot_vertices

import time
import numpy as np
import pymesh

channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Setup
PRNet.Setup()
FaceDetector.Setup()
PRNetImageCropper.Setup()

cap = cv2.VideoCapture("./IMG_0003.mov")
# cap = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/indoor_two_ppl.avi")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(frame_width)
print(frame_height)

out = None
depth_out = None
frame_num = 1490
ind = 0

useful_result = 0

while frame_num:
    frame_num -= 1
    ret, image = cap.read()
    if ret == 0:
        print("End of Video...")
        break
    if out is None:
        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
        #     'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
        out = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC(
            'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
        # depth_out = cv2.VideoWriter('output_depth.avi', cv2.VideoWriter_fourcc(
        #     'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))

    print(image.shape)

    next_request = predict_pb2.PredictRequest()
    next_request.inputs['input_image'].CopyFrom(
      tf.make_tensor_proto(image))

    face_detector = FaceDetector()
    start_time = time.time()
    face_detector.PreProcess(next_request, stub)
    face_detector.Apply()
    prnet_request = face_detector.PostProcess()
    elapsed_time = time.time() - start_time
    print('face_detector time cost: {}'.format(elapsed_time))

    if "bounding_box" in prnet_request.inputs:
        useful_result += 1

        start_time = time.time()
      	prnet_image_cropper = PRNetImageCropper()
      	prnet_image_cropper.PreProcess(prnet_request, stub)
      	prnet_image_cropper.Apply()
      	next_request = prnet_image_cropper.PostProcess()
        elapsed_time = time.time() - start_time
        print('prnet_image_cropper time cost: {}'.format(elapsed_time))

        prn = PRNet()
        start_time = time.time()
        prn.PreProcess(next_request, stub)
        prn.Apply()
        final_request = prn.PostProcess();
        elapsed_time = time.time() - start_time
        print('prnet time cost: {}'.format(elapsed_time))

        kpt = tensor_util.MakeNdarray(final_request.inputs["prnet_output"])
        vertices = tensor_util.MakeNdarray(final_request.inputs["vertices"])

        # print(vertices.shape)
        # print(image.shape)

        start_time = time.time()

        out.write(plot_vertices(np.zeros_like(image), vertices))
        # tmp = plot_vertices(np.zeros_like(image), vertices)
        # print(tmp)
        # cv2.imwrite("output/%s.jpg" % str(useful_result).zfill(3), tmp)

        elapsed_time = time.time() - start_time

        # break

    else:
        out.write(image)

print("In total, there are %d useful results." % useful_result)

