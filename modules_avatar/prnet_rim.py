import os
import numpy as np
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

class PRNet:
  @staticmethod
  def Setup():
    PRNet.resolution_inp = 256
    PRNet.resolution_op = 256
    PRNet.MaxPos = PRNet.resolution_inp * 1.1

    PRNet.uv_kpt_ind = np.loadtxt('%s/PRNet/Data/uv-data/uv_kpt_ind.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)
    PRNet.face_ind = np.loadtxt('%s/PRNet/Data/uv-data/face_ind.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)
    PRNet.triangles = np.loadtxt('%s/PRNet/Data/uv-data/triangles.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      # self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
      if (str(tensor_util.MakeNdarray(request.inputs["valid_input"])) == "False"):
        self.valid_input = False
      else:
        self.valid_input = True
        self.tform_params = tensor_util.MakeNdarray(request.inputs["tform_params"])
        self.cropped_image = tensor_util.MakeNdarray(request.inputs["cropped_image"])
    else:
      # self.image = request["client_input"]
      if (str(request["valid_input"]) == "False"):
        self.valid_input = False
      else:
        self.valid_input = True
        self.tform_params = request["tform_params"]
        self.cropped_image = request["cropped_image"]

    self.istub = istub

  def Apply(self):
    if (self.valid_input):
      image = self.cropped_image

      new_image = image[np.newaxis, :, :, :]
      new_image = new_image.astype(np.float32)

      internal_request = predict_pb2.PredictRequest()
      internal_request.model_spec.name = 'prnet_main'
      internal_request.model_spec.signature_name = 'predict_images'
      internal_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(new_image, shape=new_image.shape))

      internal_result = self.istub.Predict(internal_request, 10.0)

      pos = tensor_util.MakeNdarray(internal_result.outputs['output'])
      pos = np.squeeze(pos)
      self.cropped_pos =  pos * PRNet.MaxPos

      cropped_vertices = np.reshape(self.cropped_pos, [-1, 3]).T

      z = cropped_vertices[2, :].copy() / self.tform_params[0, 0]
      cropped_vertices[2, :] = 1
      vertices = np.dot(np.linalg.inv(self.tform_params), cropped_vertices)
      vertices = np.vstack((vertices[:2, :], z))
      pos = np.reshape(
        vertices.T, [PRNet.resolution_op, PRNet.resolution_op, 3])

      self.key_points = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
      all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
      self.vertices = all_vertices[self.face_ind, :]

    else:
      self.key_points = "None"
      self.vertices = "None"

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      # next_request.inputs["key_points"].CopyFrom(
      #   tf.make_tensor_proto(self.key_points))
      next_request.inputs["FINAL"].CopyFrom(
        tf.make_tensor_proto(self.vertices))
    else:
      next_request = dict()
      # next_request["key_points"] = self.key_points
      next_request["FINAL"] = self.vertices
    return next_request