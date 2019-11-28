import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import cv2
import tensorflow as tf

def find_face_bounding_box(boxes, scores):
  min_score_thresh = 0.7
  for i in range(0, boxes.shape[0]):
    if scores[i] > min_score_thresh:
      return tuple(boxes[i].tolist())

class FaceDetector:
  @staticmethod
  def Setup():
    pass

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub

    image_np = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    self.image_np_expanded = np.expand_dims(image_np, axis=0)
    self.frame_height, self.frame_width= self.image.shape[:2]

  def Apply(self):
    internal_request = predict_pb2.PredictRequest()
    internal_request.model_spec.name = 'face_detector'
    internal_request.model_spec.signature_name = 'predict_output'
    internal_request.inputs['image_tensor'].CopyFrom(
      tf.contrib.util.make_tensor_proto(self.image_np_expanded, shape=list(self.image_np_expanded.shape)))

    internal_result = self.istub.Predict(internal_request, 10.0)

    boxes = tensor_util.MakeNdarray(internal_result.outputs['boxes'])
    scores = tensor_util.MakeNdarray(internal_result.outputs['scores'])
    classes = tensor_util.MakeNdarray(internal_result.outputs['classes'])
    num_detections = tensor_util.MakeNdarray(internal_result.outputs['num_detections'])

    self.box = find_face_bounding_box(boxes[0], scores[0])

    if self.box is not None:
      ymin, xmin, ymax, xmax = self.box
      (left, right, top, bottom) = (xmin * self.frame_width, xmax * self.frame_width,
                                    ymin * self.frame_height, ymax * self.frame_height)
      self.normalized_box = "%s-%s-%s-%s" % (left, right, top, bottom)
    else:
      self.normalized_box = "None-None-None-None"

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["client_input"].CopyFrom(
        tf.make_tensor_proto(self.image))
      next_request.inputs['bounding_box'].CopyFrom(
        tf.make_tensor_proto(self.normalized_box))
    else:
      next_request = dict()
      next_request["client_input"] = self.image
      next_request["bounding_box"] = self.normalized_box
    return next_request
