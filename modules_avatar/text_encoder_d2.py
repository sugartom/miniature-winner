from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import sentencepiece as spm

class TextEncoder:

  # initialize static variable here
  @staticmethod
  def Setup():
    model_en = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_en.model'

    TextEncoder.sp1 = spm.SentencePieceProcessor()
    TextEncoder.sp1.Load(model_en)

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    # if (grpc_flag):
    #   image = tensor_util.MakeNdarray(request.inputs["client_input"])
    # else:
    #   image = request["client_input"]
    # data_dict["client_input"] = image

    if (grpc_flag):
      speech_recognition_output = str(tensor_util.MakeNdarray(request.inputs["speech_recognition_output"])).decode('utf-8')
    else:
      speech_recognition_output = request["speech_recognition_output"]

    data_dict["speech_recognition_output"] = speech_recognition_output

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

      # batched_data_dict["original_shape"] = []
      # for data in data_array:
      #   batched_data_dict["original_shape"].append(data["original_shape"])

      batched_data_dict["speech_recognition_output"] = []
      for data in data_array:
        batched_data_dict["speech_recognition_output"].append(data["speech_recognition_output"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      # TBA
      speech_recognition_output = batched_data_dict["speech_recognition_output"][0]
      encoded_src_list = TextEncoder.sp1.EncodeAsPieces(speech_recognition_output)
      encoded_src = ' '.join([w for w in encoded_src_list])

      batched_result_dict["encoder_output"] = [encoded_src]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1]}, {"bounding_boxes": [bb1_in_image2]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      # for i in range(batch_size):
      #   my_dict = dict()
      #   my_dict["client_input"] = batched_result_dict["client_input"][i]
      #   my_dict["original_shape"] = batched_result_dict["original_shape"][i]
      #   my_dict["num_bb"] = batched_result_dict["num_bb"][i]
      #   batched_result_array.append(my_dict)

      for i in range(batch_size):
        my_dict = dict()
        my_dict["encoder_output"] = [batched_result_dict["encoder_output"][i]]
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

    # # dr
    # result_list.append({"client_input": result_dict["client_input"], "original_shape": result_dict["original_shape"]})

    for i in range(len(result_dict["encoder_output"])):
      result_list.append({"encoder_output": result_dict["encoder_output"][i]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["encoder_output"].CopyFrom(
        tf.make_tensor_proto(result["encoder_output"].encode('utf-8')))
    else:
      next_request = dict()
      next_request["encoder_output"] = result["encoder_output"]
    return next_request
