import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
import time

from PRNet.api import PRN
from PRNet.utils.write import write_obj_with_colors

import cv2
from PRNet.utils.cv_plot import plot_kpt

from tensorflow_face_detection.utils import label_map_util

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
from tensorflow.python.framework import tensor_util


def find_face_bounding_box(boxes, scores):
    min_score_thresh = 0.7
    for i in range(0, boxes.shape[0]):
        if scores[i] > min_score_thresh:
            return tuple(boxes[i].tolist())

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU number, -1 for CPU
prn = PRN(is_dlib=False)
save_folder = 'PRNet/TestImages/results'


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './tensorflow_face_detection/protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

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


    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Expand dimensions since the model expects images to have shape:
    # [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    start_time = time.time()

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_detector'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['image_tensor'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np_expanded, shape=list(image_np_expanded.shape)))

    result = stub.Predict(request, 10.0)  # 5 seconds

    boxes = tensor_util.MakeNdarray(result.outputs['boxes'])
    scores = tensor_util.MakeNdarray(result.outputs['scores'])
    classes = tensor_util.MakeNdarray(result.outputs['classes'])
    num_detections = tensor_util.MakeNdarray(result.outputs['num_detections'])
    
    # print(boxes.shape)
    box = find_face_bounding_box(boxes[0], scores[0])
    elapsed_time = time.time() - start_time
    print('face_detector time cost: {}'.format(elapsed_time))
    if box is not None:
        ymin, xmin, ymax, xmax = box

        # print('box found: {} {} {} {}'.format(ymin, xmin, ymax, xmax))

        (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                      ymin * frame_height, ymax * frame_height)

        start_time = time.time()
        pos = prn.process(image, np.array([left, right, top, bottom])) # use dlib to detect face
        elapsed_time = time.time() - start_time
        print('prnet time cost: {}'.format(elapsed_time))
        if pos is not None:

            # -- Basic Applications
            # get landmarks
            kpt = prn.get_landmarks(pos)

            start_time = time.time()
            out.write(plot_kpt(image, kpt))
            print("capture frame count: {}".format(ind))
            elapsed_time = time.time() - start_time
            print('write video time cost: {}'.format(elapsed_time))
            ind = ind + 1
        else:
            print("pos not found")

cap.release()
out.release()
