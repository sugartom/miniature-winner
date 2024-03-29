import tensorflow as tf
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


class Jasper:

    def Setup(self):
        args_S2T = ["--config_file=/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/OpenSeq2Seq/example_configs/speech2text/jasper_10x5_8gpus_dr_mp.py",
                    "--mode=tf_serving_infer",
                    "--batch_size_per_gpu=1",
                    ]
        self.model = get_model(args_S2T, "Jasper")

        self.fetches = [
            self.model.get_data_layer().input_tensors,
            self.model.get_output_tensors(),
        ]

        self.channel = grpc.insecure_channel('0.0.0.0:8500')
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel)

    def PreProcess(self, input):
        feed_dict = self.model.get_data_layer().create_feed_dict(input)

        return feed_dict

    def Apply(self, feed_dict):
        audio = feed_dict[self.model.get_data_layer().input_tensors[
            "source_tensors"][0]]
        audio_length = feed_dict[self.model.get_data_layer().input_tensors[
            "source_tensors"][1]]
        x_id = feed_dict[self.model.get_data_layer().input_tensors[
            "source_ids"][0]]

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'jasper'
        request.model_spec.signature_name = 'predict_output'
        request.inputs['audio'].CopyFrom(
            tf.contrib.util.make_tensor_proto(audio, shape=list(audio.shape)))
        request.inputs['audio_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(audio_length,
                                              shape=list(audio_length.shape)))
        request.inputs['x_id'].CopyFrom(
            tf.contrib.util.make_tensor_proto(x_id,
                                              shape=list(x_id.shape)))

        result = self.stub.Predict.future(request, 5.0)  # 5 seconds

        inputs = self.model.get_data_layer().input_tensors
        indices_decoded_sequence = tensor_util.MakeNdarray(
            result.outputs['indices_decoded_sequence'])
        values_decoded_sequence = tensor_util.MakeNdarray(
            result.outputs['values_decoded_sequence'])
        dense_shape_decoded_sequence = tensor_util.MakeNdarray(
            result.outputs['dense_shape_decoded_sequence'])

        outputs = tf.SparseTensorValue(indices=indices_decoded_sequence,
                                       values=values_decoded_sequence,
                                       dense_shape=dense_shape_decoded_sequence)

        outputs = [outputs]

        return inputs, outputs

    def PostProcess(self, inputs, outputs):
        results = self.model.infer(inputs, outputs)

        return results[0][0]
