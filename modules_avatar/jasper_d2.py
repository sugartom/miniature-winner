from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

from OpenSeq2Seq.open_seq2seq.utils.utils import get_base_config, create_model

def get_model(args, scope):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    args, base_config, base_model, config_module = get_base_config(args)
    model = create_model(args, base_config, config_module, base_model, None)
  return model

class Jasper:

  # initialize static variable here
  @staticmethod
  def Setup():
    args_S2T = ["--config_file=/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/OpenSeq2Seq/example_configs/speech2text/jasper_10x5_8gpus_dr_mp.py",
                "--mode=tf_serving_infer",
                "--batch_size_per_gpu=1",
                ]
    Jasper.model = get_model(args_S2T, "Jasper")

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
      client_input = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      client_input = request["client_input"]
    feed_dict = Jasper.model.get_data_layer().create_feed_dict([client_input])

    data_dict["feed_dict"] = feed_dict

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

      batched_data_dict["feed_dict"] = []
      for data in data_array:
        batched_data_dict["feed_dict"].append(data["feed_dict"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      # no batching
      feed_dict = batched_data_dict["feed_dict"][0]

      audio = feed_dict[Jasper.model.get_data_layer().input_tensors["source_tensors"][0]]
      audio_length = feed_dict[Jasper.model.get_data_layer().input_tensors["source_tensors"][1]]
      x_id = feed_dict[Jasper.model.get_data_layer().input_tensors["source_ids"][0]]

      internal_request = predict_pb2.PredictRequest()
      internal_request.model_spec.name = 'jasper'
      internal_request.model_spec.signature_name = 'predict_output'
      
      internal_request.inputs['audio'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio, shape=list(audio.shape)))
      internal_request.inputs['audio_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio_length, shape=list(audio_length.shape)))
      internal_request.inputs['x_id'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_id, shape=list(x_id.shape)))

      internal_result = istub.Predict(internal_request, 10.0)  # 5 seconds

      inputs = Jasper.model.get_data_layer().input_tensors

      indices_decoded_sequence = tensor_util.MakeNdarray(
        internal_result.outputs['indices_decoded_sequence'])
      values_decoded_sequence = tensor_util.MakeNdarray(
        internal_result.outputs['values_decoded_sequence'])
      dense_shape_decoded_sequence = tensor_util.MakeNdarray(
        internal_result.outputs['dense_shape_decoded_sequence'])

      outputs = tf.SparseTensorValue(indices=indices_decoded_sequence,
                                     values=values_decoded_sequence,
                                     dense_shape=dense_shape_decoded_sequence)

      outputs = [outputs]

      results = Jasper.model.infer(inputs, outputs)
      final_result = results[0][0]

      batched_result_dict["speech_recognition_output"] = [final_result]

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
        my_dict["speech_recognition_output"] = [batched_result_dict["speech_recognition_output"][i]]
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

    for i in range(len(result_dict["speech_recognition_output"])):
      result_list.append({"speech_recognition_output": result_dict["speech_recognition_output"][i]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["speech_recognition_output"].CopyFrom(
        tf.make_tensor_proto(result["speech_recognition_output"].encode('utf-8')))
    else:
      next_request = dict()
      next_request["speech_recognition_output"] = result["speech_recognition_output"]
    return next_request
