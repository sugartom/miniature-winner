import numpy as np

from skimage.transform import estimate_transform
import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf


class PRNetImageCropper:
	resolution_inp = 256
	DST_PTS = None

	@staticmethod
	def Setup():
		PRNetImageCropper.DST_PTS = np.array([[0, 0], [0, PRNetImageCropper.resolution_inp - 1], [PRNetImageCropper.resolution_inp - 1, 0]])
		return

	def PreProcess(self, request, istub):
		self.chain_name = request.model_spec.name
		bounding_box = tensor_util.MakeNdarray(request.inputs["bounding_box"])
		self.image = tensor_util.MakeNdarray(request.inputs["input_image"])
		self.istub = istub

		left = bounding_box[0]
		right = bounding_box[1]
		top = bounding_box[2]
		bottom = bounding_box[3]
		old_size = (right - left + bottom - top) / 2
		center = np.array([right - (right - left) / 2.0,
		                   bottom - (bottom - top) / 2.0 + old_size * 0.14])
		size = int(old_size * 1.58)
		self.src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] -
		                                                                   size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
	
	def Apply(self):
		self.tform = estimate_transform('similarity', self.src_pts, PRNetImageCropper.DST_PTS)
		image = self.image / 255.
		self.cropped_image = cv2.warpAffine(image, self.tform.params[:2], dsize=(PRNetImageCropper.resolution_inp, PRNetImageCropper.resolution_inp))

	def PostProcess(self):
		next_request = predict_pb2.PredictRequest()
		next_request.inputs['cropped_image'].CopyFrom(
		  tf.make_tensor_proto(self.cropped_image))
		next_request.inputs['tform_params'].CopyFrom(
		  tf.make_tensor_proto(self.tform.params))
		next_request.inputs['input_image'].CopyFrom(
		  tf.make_tensor_proto(self.image))

		return next_request
