import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
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
        return

    def PreProcess(self, request, istub):
        self.chain_name = request.model_spec.name
        self.image = tensor_util.MakeNdarray(request.inputs["input_image"])
        self.istub = istub

        image_np = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_np_expanded = np.expand_dims(image_np, axis=0)
        self.frame_height, self.frame_width= self.image.shape[:2]

    def Apply(self):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'face_detector'
        request.model_spec.signature_name = 'predict_output'
        request.inputs['image_tensor'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.image_np_expanded, shape=list(self.image_np_expanded.shape)))

        result = self.istub.Predict(request, 10.0)  # 5 seconds

        boxes = tensor_util.MakeNdarray(result.outputs['boxes'])
        scores = tensor_util.MakeNdarray(result.outputs['scores'])
        classes = tensor_util.MakeNdarray(result.outputs['classes'])
        num_detections = tensor_util.MakeNdarray(result.outputs['num_detections'])

        self.box = find_face_bounding_box(boxes[0], scores[0])

    def PostProcess(self):
        if self.box is None:
            next_request = predict_pb2.PredictRequest()
            # Image passthrough
            next_request.inputs['input_image'].CopyFrom(
              tf.make_tensor_proto(self.image))
            return next_request

        ymin, xmin, ymax, xmax = self.box

        (left, right, top, bottom) = (xmin * self.frame_width, xmax * self.frame_width,
                                      ymin * self.frame_height, ymax * self.frame_height)
        
        # print('box found: {} {} {} {}'.format(left, right, top, bottom))

        normalized_box = np.array([left, right, top, bottom])

        next_request = predict_pb2.PredictRequest()
        # Image passthrough
        next_request.inputs['input_image'].CopyFrom(
          tf.make_tensor_proto(self.image))
        next_request.inputs['bounding_box'].CopyFrom(
            tf.make_tensor_proto(normalized_box))

        return next_request

