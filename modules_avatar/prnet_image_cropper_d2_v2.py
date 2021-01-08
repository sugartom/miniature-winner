from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import numpy as np
from skimage.transform import estimate_transform
import cv2

class PRNetImageCropper:

  # initialize static variable here
  @staticmethod
  def Setup():
    PRNetImageCropper.resolution_inp = 256
    PRNetImageCropper.DST_PTS = np.array([[0, 0], [0, PRNetImageCropper.resolution_inp - 1], [PRNetImageCropper.resolution_inp - 1, 0]])

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      raw_image = tensor_util.MakeNdarray(request.inputs["raw_image"])
      face_detector_output = str(tensor_util.MakeNdarray(request.inputs["face_detector_output"]))
      output_flag = int(tensor_util.MakeNdarray(request.inputs["output_flag"]))
    else:
      raw_image = request["raw_image"]
      face_detector_output = str(request["face_detector_output"])
      output_flag = request["output_flag"]

    bounding_box = face_detector_output.split('-')

    left = float(bounding_box[0])
    right = float(bounding_box[1])
    top = float(bounding_box[2])
    bottom = float(bounding_box[3])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0,
                       bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])

    data_dict["raw_image"] = raw_image
    data_dict["src_pts"] = src_pts
    data_dict["output_flag"] = output_flag

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
      # if (batch_size == 1):
      #   batched_data_dict["client_input"] = [data_array[0]["client_input"]]
      # else:
      #   batched_data_dict["client_input"] = data_array[0]["client_input"]
      #   for data in data_array[1:]:
      #     batched_data_dict["client_input"] = np.append(batched_data_dict["client_input"], data["client_input"], axis = 0)

      batched_data_dict["raw_image"] = []
      for data in data_array:
        batched_data_dict["raw_image"].append(data["raw_image"])

      batched_data_dict["src_pts"] = []
      for data in data_array:
        batched_data_dict["src_pts"].append(data["src_pts"])

      # output_flag
      batched_data_dict["output_flag"] = []
      for data in data_array:
        batched_data_dict["output_flag"].append(data["output_flag"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      tform = estimate_transform('similarity', batched_data_dict["src_pts"][0], PRNetImageCropper.DST_PTS)
      image = batched_data_dict["raw_image"][0] / 255.
      # print(image.shape)
      cropped_image = cv2.warpAffine(image, tform.params[:2], dsize=(PRNetImageCropper.resolution_inp, PRNetImageCropper.resolution_inp))
      # print(cropped_image.shape)

      batched_result_dict["tform_params"] = [tform.params]
      batched_result_dict["cropped_image"] = [cropped_image]
      batched_result_dict["output_flag"] = batched_data_dict["output_flag"]

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
        my_dict["output_flag"] = batched_result_dict["output_flag"][i]
        my_dict["tform_params"] = batched_result_dict["tform_params"][i]
        my_dict["cropped_image"] = batched_result_dict["cropped_image"][i]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []

    # for bb and dr, need to handle differently.

    # # bb
    # for i in range(len(result_dict[result_dict.keys()[0]])):
      # result_list.append({"client_input": result_dict["client_input"][i], "original_shape": result_dict["original_shape"][i]})

    # dr
    result_list.append({"tform_params": result_dict["tform_params"], "cropped_image": result_dict["cropped_image"], "output_flag": result_dict["output_flag"]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["tform_params"].CopyFrom(
        tf.make_tensor_proto(result["tform_params"]))
      next_request.inputs["cropped_image"].CopyFrom(
        tf.make_tensor_proto(result["cropped_image"]))
      next_request.inputs["output_flag"].CopyFrom(
        tf.make_tensor_proto(result["output_flag"]))
    else:
      next_request = dict()
      next_request["tform_params"] = result["tform_params"]
      next_request["cropped_image"] = result["cropped_image"]
      next_request["output_flag"] = result["output_flag"]
    return next_request
