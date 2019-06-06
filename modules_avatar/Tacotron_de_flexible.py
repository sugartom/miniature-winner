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
 
class Tacotron_de:

    @staticmethod
    def Setup():
        args_T2S = ["--config_file=/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/OpenSeq2Seq/example_configs/text2speech/tacotron_de_float.py",
                    "--mode=tf_serving_infer",
                    "--batch_size_per_gpu=1",
                    ]
        Tacotron_de.model = get_model(args_T2S, "T2S")

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["decoder_output"]))
        else:
            self.request_input = request_input["decoder_output"]
        
        self.istub = istub
        self.feed_dict = Tacotron_de.model.get_data_layer().create_feed_dict([self.request_input])

    def Apply(self):

        text = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][0]]
        text_length = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][1]]

        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = 'tacotron_de'
        internal_request.model_spec.signature_name = 'predict_output'

        internal_request.inputs['text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(text, shape=list(text.shape)))
        internal_request.inputs['text_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(text_length, shape=list(text_length.shape)))

        internal_result = self.istub.Predict(internal_request, 10.0)  # 5 seconds

        self.audio_length = np.array(
            internal_result.outputs['audio_length'].int_val)

        self.audio = tensor_util.MakeNdarray(
            internal_result.outputs['audio_prediction'])

        results = Tacotron_de.model.infer(self.feed_dict, [None, None, None, None, self.audio_length, self.audio])

        audio_length = results[1][4][0]

        if Tacotron_de.model.get_data_layer()._both:
            prediction = results[1][5][0]

        else:
            prediction = results[1][1][0]

        prediction = prediction[:audio_length - 1, :]
        self.final_result = Tacotron_de.model.get_data_layer().get_magnitude_spec(prediction)


    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            # tt = self.final_result.encode('utf-8')
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['FINAL'].CopyFrom(
              tf.make_tensor_proto(self.final_result, shape=self.final_result.shape))

            return next_request
        else:
            result = dict()
            result["FINAL"] = self.final_result
            return result
