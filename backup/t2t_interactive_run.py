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

import sentencepiece as spm

args_T2T = ["--config_file=example_configs/text2text/en-de/transformer-bp-fp32.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/transformer-base/Transformer-FP32-H-256",
            "--batch_size_per_gpu=1",
            ]
# A simpler version of what run.py does. It returns the created model and
# its saved checkpoint

model_en = '/home/oscar/sdb3/data/wmt16_de_en/m_en.model'
model_de = '/home/oscar/sdb3/data/wmt16_de_en/m_de.model'


def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint

model_T2T, checkpoint_T2T = get_model(args_T2T, "T2T")

# # Create the session and load the checkpoints
# sess_config = tf.ConfigProto(allow_soft_placement=True)
# sess_config.gpu_options.allow_growth = True
# sess = tf.Session(config=sess_config)
# vars_T2T = {}
# for v in tf.get_collection(tf.GraphKeys.VARIABLES):
#     if "T2T" in v.name:
#         vars_T2T["/".join(v.op.name.split("/")[1:])] = v

# saver_T2T = tf.train.Saver(vars_T2T)
# saver_T2T.restore(sess, checkpoint_T2T)


sp1 = spm.SentencePieceProcessor()
sp1.Load(model_en)
sp2 = spm.SentencePieceProcessor()
sp2.Load(model_de)


def get_interactive_infer_results(model, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    src_text = feed_dict[model.get_data_layer().input_tensors["source_tensors"][0]]
    src_text_length = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][1]].astype(np.int32)
    print(src_text)
    print(src_text_length.dtype)

    channel = grpc.insecure_channel('0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'text2text'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['src_text'].CopyFrom(
        tf.contrib.util.make_tensor_proto(src_text, shape=list(src_text.shape)))
    request.inputs['src_text_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(src_text_length, shape=list(src_text_length.shape)))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print('Result returned from rpc')

    tgt_txt = tensor_util.MakeNdarray(
        result_future.result().outputs['tgt_txt'])

    inputs = {"source_tensors" : [src_text, src_text_length]}
    outputs = [tgt_txt]

    # inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    # export_path = "/tmp/text2text/0"
    # print('Exporting trained model to', export_path)

    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # # Define input tensors
    # src_text = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][0])
    # src_text_length = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][1])

    # tgt_txt = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[0])

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'src_text': src_text, 'src_text_length': src_text_length},
    #         outputs={'tgt_txt': tgt_txt},
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

    return model.infer(inputs, outputs)


def infer(line):
    print(line)

    # Encode sentence
    encoded_src_list = sp1.EncodeAsPieces(line)
    encoded_src = ' '.join([w for w in encoded_src_list])

    # Generate translation
    results = get_interactive_infer_results(model_T2T, model_in=[encoded_src])
    encoded_tgt = results[1][0]
    decoded_tgt = sp2.DecodePieces(encoded_tgt.split(" "))

    print("Translation")
    print(decoded_tgt)


while True:
    print("Input English")

    line = input()
    if line == "":
        break
    infer(line)

    print("Infer complete")
