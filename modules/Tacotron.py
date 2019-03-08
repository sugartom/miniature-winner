import tensorflow as tf
import numpy as np
from OpenSeq2Seq.open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util


def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint

 
class Tacotron:

    def Setup(self):
        args_T2S = ["--config_file=OpenSeq2Seq/example_configs/text2speech/tacotron_LJ_float.py",
                    "--mode=interactive_infer",
                    "--logdir=checkpoints/tacotron-LJ-float/checkpoint/",
                    "--batch_size_per_gpu=1",
                    ]
        self.model, checkpoint_T2S = get_model(args_T2S, "T2S")

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

        text = feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][0]]
        text_length = feed_dict[self.model.get_data_layer().input_tensors[
            "source_tensors"][1]]

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'saved_models'
        request.model_spec.signature_name = 'predict_output'
        request.inputs['text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(text, shape=list(text.shape)))
        request.inputs['text_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(text_length, shape=list(text_length.shape)))

        result_future = self.stub.Predict.future(request, 5.0)  # 5 seconds
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            print('Result returned from rpc')

        audio_length = np.array(
            result_future.result().outputs['audio_length'].int_val)

        audio = tensor_util.MakeNdarray(
            result_future.result().outputs['audio_prediction'])
        
        inputs = feed_dict
        outputs = [None, None, None, None, audio_length, audio]

        return inputs, outputs

    def PostProcess(self, inputs, outputs):
        results = self.model.infer(inputs, outputs)

        audio_length = results[1][4][0]

        if self.model.get_data_layer()._both:
            prediction = results[1][5][0]

        else:
            prediction = results[1][1][0]

        prediction = prediction[:audio_length - 1, :]
        mag_prediction = self.model.get_data_layer().get_magnitude_spec(prediction)

        return mag_prediction

