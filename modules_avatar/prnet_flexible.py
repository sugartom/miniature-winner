import numpy as np
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import time

# import pickle

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')
from PRNet.utils.cv_plot import plot_vertices

class PRNet:

    @staticmethod
    def Setup():
        PRNet.resolution_inp = 256
        PRNet.resolution_op = 256
        PRNet.MaxPos = PRNet.resolution_inp * 1.1
        prefix='/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/'

        PRNet.uv_kpt_ind = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        PRNet.face_ind = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/face_ind.txt').astype(np.int32)
        PRNet.triangles = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/triangles.txt').astype(np.int32)

    def PreProcess(self, request_input, istub, grpc_flag):
        self.istub = istub
        self.valid_input = False

        if (grpc_flag):
          self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
          self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
          if (str(tensor_util.MakeNdarray(request_input.inputs["valid_input"])) == "False"):
            return
          else:
            self.valid_input = True
            # self.tform_params = pickle.loads(tensor_util.MakeNdarray(request_input.inputs["tform_params"]))
            # self.cropped_image = pickle.loads(tensor_util.MakeNdarray(request_input.inputs["cropped_image"]))
            self.tform_params = tensor_util.MakeNdarray(request_input.inputs["tform_params"])
            self.cropped_image = tensor_util.MakeNdarray(request_input.inputs["cropped_image"])
        else:
          self.image = request_input["client_input"]
          if (str(request_input["valid_input"]) == "False"):
            return
          else:
            self.valid_input = True
            self.tform_params = request_input["tform_params"]
            self.cropped_image = request_input["cropped_image"]

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

            # self.output = plot_vertices(np.zeros_like(self.image), vertices)

        else:
            # self.output = self.image
            self.key_points = "None"
            self.vertices = "None"

    def PostProcess(self, grpc_flag):
        # Unused output, maybe later...?
        # next_request.inputs['prnet_output'].CopyFrom(
          # tf.make_tensor_proto(key_points))

        if (grpc_flag):
            next_request = predict_pb2.PredictRequest()
            # next_request.inputs["FINAL"].CopyFrom(
            #   tf.make_tensor_proto(cv2.imencode('.jpg', self.output)[1].tostring()))
            # next_request.inputs["FINAL"].CopyFrom(
            #   tf.make_tensor_proto(pickle.dumps(self.key_points)))
            # next_request.inputs["vertices"].CopyFrom(
            #   tf.make_tensor_proto(pickle.dumps(self.vertices)))
            next_request.inputs["FINAL"].CopyFrom(
              tf.make_tensor_proto(self.key_points))
            next_request.inputs["vertices"].CopyFrom(
              tf.make_tensor_proto(self.vertices))
            return next_request
        else:
            result = dict()
            result["FINAL"] = self.key_points
            result["vertices"] = self.vertices
            return result
