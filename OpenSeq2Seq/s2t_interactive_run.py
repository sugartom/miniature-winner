import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import tensor_util

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
from open_seq2seq.models.text2speech import save_audio

args_S2T = ["--config_file=example_configs/speech2text/ds2_large_8gpus_mp.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/deepspeech2/ds2_large/",
            "--batch_size_per_gpu=1",
            ]
# A simpler version of what run.py does. It returns the created model and
# its saved checkpoint


def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint

model_S2T, checkpoint_S2T = get_model(args_S2T, "S2T")

# vars_S2T = {}
# for v in tf.get_collection(tf.GraphKeys.VARIABLES):
#     if "S2T" in v.name:
#         vars_S2T["/".join(v.op.name.split("/")[1:])] = v

# saver_S2T = tf.train.Saver(vars_S2T)
# saver_S2T.restore(sess, checkpoint_S2T)


def get_interactive_infer_results(model, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    # inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    # export_path = "/tmp/speech2text/0"
    # print('Exporting trained model to', export_path)

    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # # Define input tensors
    # audio = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][0])
    # audio_length = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][1])
    # x_id = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_ids"][0])

    # # Define output tensors
    # # decoded_sequence = tf.saved_model.utils.build_tensor_info(
    # #     model.get_output_tensors()[0])

    # # prediction_signature = (
    # #     tf.saved_model.signature_def_utils.build_signature_def(
    # #         inputs={'audio': audio, 'audio_length': audio_length, 'x_id': x_id},
    # #         outputs={'decoded_sequence': decoded_sequence},
    # #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # indices_decoded_sequence = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[0].indices)
    # values_decoded_sequence = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[0].values)
    # dense_shape_decoded_sequence = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[0].dense_shape)

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'audio': audio, 'audio_length': audio_length, 'x_id': x_id},
    #         outputs={'indices_decoded_sequence': indices_decoded_sequence,
    #                  'values_decoded_sequence': values_decoded_sequence,
    #                  'dense_shape_decoded_sequence': dense_shape_decoded_sequence},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'predict_output':
    #             prediction_signature,
    #     },
    #     main_op=tf.tables_initializer(),
    #     strip_default_attrs=True)

    # builder.save()

    audio = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][0]]
    audio_length = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][1]]
    x_id = feed_dict[model.get_data_layer().input_tensors[
        "source_ids"][0]]

    print('audio shape: ', audio.shape)
    print('audio_length shape: ', audio_length.shape)

    # inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    channel = grpc.insecure_channel('0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'speech2text'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['audio'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio, shape=list(audio.shape)))
    request.inputs['audio_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio_length,
                                          shape=list(audio_length.shape)))
    request.inputs['x_id'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_id,
                                          shape=list(x_id.shape)))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print('Result returned from rpc')

    inputs = model.get_data_layer().input_tensors
    indices_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['indices_decoded_sequence'])
    values_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['values_decoded_sequence'])
    dense_shape_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['dense_shape_decoded_sequence'])

    outputs = tf.SparseTensorValue(indices=indices_decoded_sequence,
                                   values=values_decoded_sequence,
                                   dense_shape=dense_shape_decoded_sequence)

    outputs = [outputs]

    return model.infer(inputs, outputs)


def infer(line):
    print(line)

    # Generate speech
    results = get_interactive_infer_results(model_S2T, model_in=[line])
    english_recognized = results[0][0]

    print("Recognized Speech")
    print(english_recognized)


while True:
    print("Input path")

    line = input()
    if line == "":
        break
    infer(line)

    print("Infer complete")
