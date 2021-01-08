import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import cv2
import tensorflow as tf

def find_face_bounding_box(boxes, scores):
    min_score_thresh = 0.2
    for i in range(0, boxes.shape[0]):
        if scores[i] > min_score_thresh:
            return tuple(boxes[i].tolist())

class FaceDetector:
    @staticmethod
    def Setup():
        return

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
            self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        else:
            self.image = request_input["client_input"]

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
            try:
                self.request_input
            except AttributeError:
                self.request_input = cv2.imencode('.jpg', self.image)[1].tostring()

            next_request = predict_pb2.PredictRequest()
            next_request.inputs['client_input'].CopyFrom(
              tf.make_tensor_proto(self.request_input))
            next_request.inputs['bounding_box'].CopyFrom(
              tf.make_tensor_proto(self.normalized_box))
            return next_request

        else:
            result = dict()
            result["client_input"] = self.image
            result["bounding_box"] = self.normalized_box
            return result
