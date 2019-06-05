import numpy as np

from skimage.transform import estimate_transform
import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import pickle

class PRNetImageCropper:

  @staticmethod
  def Setup():
    PRNetImageCropper.resolution_inp = 256
    PRNetImageCropper.DST_PTS = np.array([[0, 0], [0, PRNetImageCropper.resolution_inp - 1], [PRNetImageCropper.resolution_inp - 1, 0]])
    return

  def PreProcess(self, request_input, istub, grpc_flag):
    if (grpc_flag):
      self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
      self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
      bounding_box = str(tensor_util.MakeNdarray(request_input.inputs["bounding_box"]))
    else:
      self.image = request_input["client_input"]
      bounding_box = request_input["bounding_box"]

    self.istub = istub

    bounding_box = bounding_box.split('-')

    if (bounding_box[0] == "None"):
      self.valid_input = False
    else:
      self.valid_input = True
      left = float(bounding_box[0])
      right = float(bounding_box[1])
      top = float(bounding_box[2])
      bottom = float(bounding_box[3])
      old_size = (right - left + bottom - top) / 2
      center = np.array([right - (right - left) / 2.0,
                         bottom - (bottom - top) / 2.0 + old_size * 0.14])
      size = int(old_size * 1.58)
      self.src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
  
  def Apply(self):
    if (self.valid_input):
      self.tform = estimate_transform('similarity', self.src_pts, PRNetImageCropper.DST_PTS)
      image = self.image / 255.
      self.cropped_image = cv2.warpAffine(image, self.tform.params[:2], dsize=(PRNetImageCropper.resolution_inp, PRNetImageCropper.resolution_inp))

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      try:
        self.request_input
      except AttributeError:
        self.request_input = cv2.imencode('.jpg', self.image)[1].tostring()
      
      if (self.valid_input):
        cropped_image_output = pickle.dumps(self.cropped_image)
        tform_params_output = pickle.dumps(self.tform.params)
        valid_input_output = "True"
      else:
        cropped_image_output = "None"
        tform_params_output = "None"
        valid_input_output = "False"

      next_request = predict_pb2.PredictRequest()
      next_request.inputs['client_input'].CopyFrom(
        tf.make_tensor_proto(self.request_input))
      next_request.inputs['cropped_image'].CopyFrom(
        tf.make_tensor_proto(cropped_image_output))
      next_request.inputs['tform_params'].CopyFrom(
        tf.make_tensor_proto(tform_params_output))
      next_request.inputs['valid_input'].CopyFrom(
        tf.make_tensor_proto(valid_input_output))

      return next_request

    else:

      if (self.valid_input):
        cropped_image_output = self.cropped_image
        tform_params_output = self.tform.params
        valid_input_output = "True"
      else:
        cropped_image_output = "None"
        tform_params_output = "None"
        valid_input_output = "False"

      result = dict()
      result["client_input"] = self.image
      result["cropped_image"] = cropped_image_output
      result["tform_params"] = tform_params_output
      result["valid_input"] = valid_input_output

      return result
