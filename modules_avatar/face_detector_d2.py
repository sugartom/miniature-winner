from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import numpy as np
import cv2

def find_face_bounding_box(boxes, scores):
  min_score_thresh = 0.7
  for i in range(0, boxes.shape[0]):
    if scores[i] > min_score_thresh:
      return tuple(boxes[i].tolist())

class FaceDetector:

  # initialize static variable here
  @staticmethod
  def Setup():
    pass

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      raw_image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      raw_image = request["client_input"]

    image_np = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    frame_height, frame_width = raw_image.shape[:2]

    data_dict["raw_image"] = raw_image
    data_dict["image_np_expanded"] = image_np_expanded
    data_dict["frame_height"] = frame_height
    data_dict["frame_width"] = frame_width

    return data_dict

  # for an array of requests from a batch, convert them to a dict,
  # where each key has a lit of values
  # input: data_array = [{"image": image1, "meta": meta1}, {"image": image2, "meta": meta2}]
  # output: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None
    else:
      batched_data_dict = dict()

      # for each key in data_array[0], convert it to batched_data_dict[key][]
      batched_data_dict["image_np_expanded"] = data_array[0]["image_np_expanded"]
      for data in data_array[1:]:
        batched_data_dict["image_np_expanded"] = np.append(batched_data_dict["image_np_expanded"], data["image_np_expanded"], axis = 0)

      batched_data_dict["raw_image"] = []
      for data in data_array:
        batched_data_dict["raw_image"].append(data["raw_image"])

      batched_data_dict["frame_height"] = []
      for data in data_array:
        batched_data_dict["frame_height"].append(data["frame_height"])

      batched_data_dict["frame_width"] = []
      for data in data_array:
        batched_data_dict["frame_width"].append(data["frame_width"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'face_detector'
      request.model_spec.signature_name = 'predict_output'
      request.inputs['image_tensor'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["image_np_expanded"], shape = batched_data_dict["image_np_expanded"].shape))

      result = istub.Predict(request, 10.0)

      boxes = tensor_util.MakeNdarray(result.outputs['boxes'])
      scores = tensor_util.MakeNdarray(result.outputs['scores'])
      classes = tensor_util.MakeNdarray(result.outputs['classes'])
      num_detections = tensor_util.MakeNdarray(result.outputs['num_detections'])

      face_detector_output = []

      for i in range(len(boxes)):
        box = find_face_bounding_box(boxes[i], scores[i])

        if (box is not None):
          ymin, xmin, ymax, xmax = box
          (left, right, top, bottom) = (xmin * batched_data_dict["frame_width"][i], xmax * batched_data_dict["frame_width"][i], ymin * batched_data_dict["frame_height"][i], ymax * batched_data_dict["frame_height"][i])
          normalized_box = "%s-%s-%s-%s" % (left, right, top, bottom)
        else:
          normalized_box = None

        face_detector_output.append(normalized_box)

      batched_result_dict["face_detector_output"] = face_detector_output
      batched_result_dict["raw_image"] = batched_data_dict["raw_image"]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1]}, {"bounding_boxes": [bb1_in_image2]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        my_dict = dict()
        my_dict["raw_image"] = batched_result_dict["raw_image"][i]
        my_dict["face_detector_output"] = batched_result_dict["face_detector_output"][i]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []

    # for bb and dr, need to handle differently.

    # # bb
    # for i in range(len(result_dict["face_detector_output"])):
    #   if (result_dict["face_detector_output"][i] is not None):
    #     result_list.append({"raw_image": result_dict["raw_image"][i], "face_detector_output": result_dict["face_detector_output"][i]})

    # # dr
    if (result_dict["face_detector_output"] is not None):
      result_list.append({"raw_image": result_dict["raw_image"], "face_detector_output": result_dict["face_detector_output"]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["raw_image"].CopyFrom(
        tf.make_tensor_proto(result["raw_image"]))
      next_request.inputs["face_detector_output"].CopyFrom(
        tf.make_tensor_proto(result["face_detector_output"]))
    else:
      next_request = dict()
      next_request["raw_image"] = result["raw_image"]
      next_request["face_detector_output"] = result["face_detector_output"]
    return next_request
