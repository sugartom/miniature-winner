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
from PRNet.utils.cv_plot import plot_kpt

import time

channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Setup
PRNet.Setup()
FaceDetector.Setup()
PRNetImageCropper.Setup()


cap = cv2.VideoCapture("./tensorflow_face_detection/media/test.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = None
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

    next_request = predict_pb2.PredictRequest()
    next_request.inputs['input_image'].CopyFrom(
      tf.make_tensor_proto(image))

    face_detector = FaceDetector()
    start_time = time.time()
    face_detector.PreProcess(next_request, stub)
    face_detector.Apply()
    prnet_request = face_detector.PostProcess()
    elapsed_time = time.time() - start_time

    if "bounding_box" in prnet_request.inputs:
    	prnet_image_cropper = PRNetImageCropper()
    	prnet_image_cropper.PreProcess(prnet_request, stub)
    	prnet_image_cropper.Apply()
    	next_request = prnet_image_cropper.PostProcess();


        prn = PRNet()
        start_time = time.time()
        prn.PreProcess(next_request, stub)
        elapsed_time = time.time() - start_time
        print('prnet time cost: {}'.format(elapsed_time))
        prn.Apply()
        final_request = prn.PostProcess();


        kpt = tensor_util.MakeNdarray(final_request.inputs["prnet_output"])
        out.write(plot_kpt(image, kpt))
    else:
        out.write(image)

