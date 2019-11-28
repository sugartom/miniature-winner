import numpy as np
from skimage.transform import estimate_transform
import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

class PRNetImageCropper:
  @staticmethod
  def Setup():
    PRNetImageCropper.resolution_inp = 256
    PRNetImageCropper.DST_PTS = np.array([[0, 0], [0, PRNetImageCropper.resolution_inp - 1], [PRNetImageCropper.resolution_inp - 1, 0]])
    return

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
      bounding_box = str(tensor_util.MakeNdarray(request.inputs["bounding_box"]))
    else:
      self.image = request["client_input"]
      bounding_box = request["bounding_box"]

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
    if (self.valid_input):
      cropped_image_output = self.cropped_image
      tform_params_output = self.tform.params
      valid_input_output = "True"
    else:
      cropped_image_output = "None"
      tform_params_output = "None"
      valid_input_output = "False"

    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      # next_request.inputs['client_input'].CopyFrom(
      #   tf.make_tensor_proto(self.image))
      next_request.inputs['cropped_image'].CopyFrom(
        tf.make_tensor_proto(cropped_image_output))
      next_request.inputs['tform_params'].CopyFrom(
        tf.make_tensor_proto(tform_params_output))
      next_request.inputs['valid_input'].CopyFrom(
        tf.make_tensor_proto(valid_input_output))
    else:
      next_request = dict()
      # next_request["client_input"] = self.image
      next_request["cropped_image"] = cropped_image_output
      next_request["tform_params"] = tform_params_output
      next_request["valid_input"] = valid_input_output
    return next_request