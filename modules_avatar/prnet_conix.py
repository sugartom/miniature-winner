import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import time

class PRNet:
    resolution_inp = 256
    resolution_op = 256
    MaxPos = resolution_inp * 1.1
    # DST_PTS = np.array(
    #     [[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    uv_kpt_ind = None
    face_ind = None
    triangles = None

    @staticmethod
    def Setup():
        prefix='/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/'

        PRNet.uv_kpt_ind = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        PRNet.face_ind = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/face_ind.txt').astype(np.int32)
        PRNet.triangles = np.loadtxt(
            prefix + '/PRNet/Data/uv-data/triangles.txt').astype(np.int32)

    def PreProcess(self, request, istub):
      self.chain_name = request.model_spec.name
      self.image = tensor_util.MakeNdarray(request.inputs["input_image"])
      self.tform_params = tensor_util.MakeNdarray(request.inputs["tform_params"])
      self.cropped_image = tensor_util.MakeNdarray(request.inputs["cropped_image"])
      self.istub = istub

    def Apply(self):
        image = self.cropped_image

        new_image = image[np.newaxis, :, :, :]
        new_image = new_image.astype(np.float32)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'prnet_main'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(new_image, shape=new_image.shape))
        result = self.istub.Predict(request, 10.0)  # 10 secs timeout

        pos = tensor_util.MakeNdarray(result.outputs['output'])

        pos = np.squeeze(pos)

        self.cropped_pos =  pos * PRNet.MaxPos

    def PostProcess(self):
        cropped_vertices = np.reshape(self.cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / self.tform_params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(self.tform_params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(
            vertices.T, [PRNet.resolution_op, PRNet.resolution_op, 3])

        key_points = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        # next_request = predict_pb2.PredictRequest()
        # next_request.inputs['prnet_output'].CopyFrom(
        #   tf.make_tensor_proto(key_points))
        # next_request.inputs['vertices'].CopyFrom(
        #   tf.make_tensor_proto(vertices))

        # return next_request
        return key_points, vertices