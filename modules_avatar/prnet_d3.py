from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import os
import numpy as np
import cv2

class PRNet:

  # initialize static variable here
  @staticmethod
  def Setup():
    PRNet.resolution_inp = 256
    PRNet.resolution_op = 256
    PRNet.MaxPos = PRNet.resolution_inp * 1.1

    PRNet.uv_kpt_ind = np.loadtxt('%s/PRNet/Data/uv-data/uv_kpt_ind.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)
    PRNet.face_ind = np.loadtxt('%s/PRNet/Data/uv-data/face_ind.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)
    PRNet.triangles = np.loadtxt('%s/PRNet/Data/uv-data/triangles.txt' % os.environ['MINIATURE_WINNER_PATH']).astype(np.int32)

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      tform_params = tensor_util.MakeNdarray(request.inputs["tform_params"])
      cropped_image = tensor_util.MakeNdarray(request.inputs["cropped_image"])
      output_flag = int(tensor_util.MakeNdarray(request.inputs["output_flag"]))
    else:
      tform_params = request["tform_params"]
      cropped_image = request["cropped_image"]
      output_flag = request["output_flag"]

    new_image = cropped_image[np.newaxis, :, :, :]
    new_image = new_image.astype(np.float32)

    data_dict["tform_params"] = tform_params
    data_dict["new_image"] = new_image
    data_dict["output_flag"] = output_flag

    return data_dict

  # for an array of requests from a batch, convert them to a dict,
  # where each key has a lit of values
  # input: data_array = [{"client_input": image1, "meta": meta1, "raw_image": raw_image1}, 
  #                      {"client_input": image2, "meta": meta2, "raw_image": raw_image2}]
  # output: batched_data_dict = {"client_input": batched_image, 
  #                              "meta": [meta1, meta2], 
  #                              "raw_image": [raw_image1, raw_image2]}
  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None
    else:
      batched_data_dict = dict()

      batched_data_dict["new_image"] = data_array[0]["new_image"]
      for data in data_array[1:]:
        batched_data_dict["new_image"] = np.append(batched_data_dict["new_image"], data["new_image"], axis = 0)

      batched_data_dict["tform_params"] = []
      for data in data_array:
        batched_data_dict["tform_params"].append(data["tform_params"])

      # output_flag
      batched_data_dict["output_flag"] = []
      for data in data_array:
        batched_data_dict["output_flag"].append(data["output_flag"])

      return batched_data_dict

  # input: batched_data_dict = {"client_input": batched_image, 
  #                             "meta": [meta1, meta2], 
  #                             "raw_image": [raw_image1, raw_image2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]], 
  #                                "meta": [meta1, meta2], 
  #                                "raw_image": [raw_image1, raw_image2]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict["output_flag"])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'prnet_main'
      request.model_spec.signature_name = 'predict_images'
      request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["new_image"], shape = batched_data_dict["new_image"].shape))

      result = istub.Predict(request, 10.0)

      pos_batched = tensor_util.MakeNdarray(result.outputs['output'])

      vertices_array = []

      for i in range(batch_size):
        pos = pos_batched[i, :, :, :]
        pos = np.squeeze(pos)

        cropped_pos = pos * PRNet.MaxPos

        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T

        z = cropped_vertices[2, :].copy() / batched_data_dict["tform_params"][i][0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(batched_data_dict["tform_params"][i]), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(
          vertices.T, [PRNet.resolution_op, PRNet.resolution_op, 3])

        all_vertices = np.reshape(pos, [PRNet.resolution_op**2, -1])
        vertices = all_vertices[PRNet.face_ind, :]

        vertices_array.append(vertices)

      batched_result_dict["vertices"] = vertices_array
      batched_result_dict["output_flag"] = batched_data_dict["output_flag"]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]], 
  #                               "meta": [meta1, meta2], 
  #                               "raw_image": [raw_image1, raw_image2]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1], "meta": meta1, "raw_image": raw_image1, "output_flag": [1, 0]}, 
  #                                 {"bounding_boxes": [bb1_in_image2], "meta": meta2, "raw_image": raw_image2, "output_flag": [1]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict["output_flag"])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        result_dict = dict()

        output_flag = batched_result_dict["output_flag"][i]

        if (output_flag == 1):
          result_dict["vertices"] = batched_result_dict["vertices"][i]
        else:
          result_dict["vertices"] = None

        batched_result_array.append(result_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1], 
  #                       "meta": meta1, 
  #                       "raw_image": raw_image1}
  # output: result_list = [{"bounding_boxes": bb1_in_image1, "meta": meta1, "raw_image": raw_image1}, 
  #                        {"bounding_boxes": bb2_in_image1, "meta": meta1, "raw_image": raw_image1}]
  def GetResultList(self, result_dict):
    result_list = []

    # for i in range(len(result_dict["bounding_boxes"])):
      # result_list.append({"bounding_boxes": result_dict["bounding_boxes"][i], "original_shape": result_dict["original_shape"], "raw_image": result_dict["raw_image"]})

    if (result_dict["vertices"] is not None):
      result_list.append({"vertices": result_dict["vertices"]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1, 
  #                  "meta": meta1, 
  #                  "raw_image": raw_image1,
  #                  "output_flag": 1}
  # output: next_request["bounding_boxes"] = bb1_in_image1
  #         next_request["meta"] = meta1
  #         next_request["raw_image"] = raw_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["vertices"].CopyFrom(
        tf.make_tensor_proto(result["vertices"]))
    else:
      next_request = dict()
      next_request["vertices"] = result["vertices"]
    return next_request
