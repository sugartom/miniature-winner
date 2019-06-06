import tensorflow as tf
import numpy as np
from OpenSeq2Seq.open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util

def get_model(args, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        args, base_config, base_model, config_module = get_base_config(args)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model


class Transformer:

    @staticmethod
    def Setup():
        args_T2T = ["--config_file=/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/OpenSeq2Seq/example_configs/text2text/en-de/transformer-bp-fp32.py",
                    "--mode=tf_serving_infer",
                    "--batch_size_per_gpu=1",
                    ]
        Transformer.model = get_model(args_T2T, "T2T")

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request.inputs["encoder_output"])).decode('utf-8')
        else:
            self.request_input = request_input["encoder_output"]

        self.istub = istub
        self.feed_dict = Transformer.model.get_data_layer().create_feed_dict([self.request_input])

    def Apply(self):
        src_text = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][0]]
        src_text_length = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][1]].astype(np.int32)

        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = 'transformer'
        internal_request.model_spec.signature_name = 'predict_output'

        internal_request.inputs['src_text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(src_text, shape=list(src_text.shape)))
        internal_request.inputs['src_text_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(src_text_length, shape=list(src_text_length.shape)))

        internal_result = self.istub.Predict(internal_request, 10.0)  # 5 seconds

        tgt_txt = tensor_util.MakeNdarray(
            internal_result.outputs['tgt_txt'])

        self.inputs = {"source_tensors" : [src_text, src_text_length]}
        self.outputs = [tgt_txt]

        result = self.model.infer(self.inputs, self.outputs)
        self.final_result = result[1][0]

    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            tt = self.final_result.encode('utf-8')
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['general_transformer_output'].CopyFrom(
              tf.make_tensor_proto(tt))

            return next_request
        else:
            result = dict()
            result["general_transformer_output"] = self.final_result
            return result