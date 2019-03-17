import tensorflow as tf
import numpy as np
from OpenSeq2Seq.open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

def get_model(args, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        args, base_config, base_model, config_module = get_base_config(args)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model


class Convs2s:

    def Setup(self):
        args_T2T = ["--config_file=OpenSeq2Seq/example_configs/text2text/en-de/en-de-convs2s-8-gpu.py",
                    "--mode=tf_serving_infer",
                    "--batch_size_per_gpu=1",
                    ]
        self.model = get_model(args_T2T, "transformer-big")

        self.fetches = [
            self.model.get_data_layer().input_tensors,
            self.model.get_output_tensors(),
        ]

        self.channel = grpc.insecure_channel('0.0.0.0:8500')
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def PreProcess(self, input):
        feed_dict = self.model.get_data_layer().create_feed_dict(input)

        return feed_dict

    def Apply(self, feed_dict):
        src_text = feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][0]]
        src_text_length = feed_dict[self.model.get_data_layer().input_tensors[
            "source_tensors"][1]].astype(np.int32)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'conv_s2s'
        request.model_spec.signature_name = 'predict_output'
        request.inputs['src_text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(src_text, shape=list(src_text.shape)))
        request.inputs['src_text_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(src_text_length, shape=list(src_text_length.shape)))

        result_future = self.stub.Predict.future(request, 5.0)  # 5 seconds
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            print('Result returned from rpc')

        tgt_txt = tensor_util.MakeNdarray(
            result_future.result().outputs['tgt_txt'])

        inputs = {"source_tensors" : [src_text, src_text_length]}
        outputs = [tgt_txt]

        return inputs, outputs

    def PostProcess(self, inputs, outputs):
        result = self.model.infer(inputs, outputs)

        return result[1][0]
