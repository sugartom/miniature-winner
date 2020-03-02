import tensorflow as tf
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

class Wave2Letter:

    @staticmethod
    def Setup():
        args_S2T = ["--config_file=/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/OpenSeq2Seq/example_configs/speech2text/w2lplus_large_8gpus_mp.py",
                    "--mode=tf_serving_infer",
                    "--batch_size_per_gpu=1",
                    ]
        Wave2Letter.model = get_model(args_S2T, "Wave2Letter")

    def PreProcess(self, request, istub, grpc_flag):
        if (grpc_flag):
            self.request = tensor_util.MakeNdarray(request.inputs["client_input"])
        else:
            self.request = request["client_input"]
        self.istub = istub

        self.feed_dict = Wave2Letter.model.get_data_layer().create_feed_dict([self.request])

    def Apply(self):
        audio = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][0]]
        audio_length = self.feed_dict[self.model.get_data_layer().input_tensors["source_tensors"][1]]
        x_id = self.feed_dict[self.model.get_data_layer().input_tensors["source_ids"][0]]

        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = 'wave2letter'
        internal_request.model_spec.signature_name = 'predict_output'
        
        internal_request.inputs['audio'].CopyFrom(
            tf.contrib.util.make_tensor_proto(audio, shape=list(audio.shape)))
        internal_request.inputs['audio_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(audio_length, shape=list(audio_length.shape)))
        internal_request.inputs['x_id'].CopyFrom(
            tf.contrib.util.make_tensor_proto(x_id, shape=list(x_id.shape)))

        internal_result = self.istub.Predict(internal_request, 10.0)  # 5 seconds

        self.inputs = Wave2Letter.model.get_data_layer().input_tensors

        indices_decoded_sequence = tensor_util.MakeNdarray(
            internal_result.outputs['indices_decoded_sequence'])
        values_decoded_sequence = tensor_util.MakeNdarray(
            internal_result.outputs['values_decoded_sequence'])
        dense_shape_decoded_sequence = tensor_util.MakeNdarray(
            internal_result.outputs['dense_shape_decoded_sequence'])

        outputs = tf.SparseTensorValue(indices=indices_decoded_sequence,
                                       values=values_decoded_sequence,
                                       dense_shape=dense_shape_decoded_sequence)

        self.outputs = [outputs]

        results = Wave2Letter.model.infer(self.inputs, self.outputs)
        self.final_result = results[0][0]

    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            tt = self.final_result.encode('utf-8')
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['speech_recognition_output'].CopyFrom(
              tf.make_tensor_proto(tt))

            return next_request
        else:
            result = dict()
            result["speech_recognition_output"] = self.final_result
            return result