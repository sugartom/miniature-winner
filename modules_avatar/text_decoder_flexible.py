import sentencepiece as spm

import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2

class TextDecoder:

    @staticmethod
    def Setup():
        model_de = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_de.model'

        TextDecoder.sp1 = spm.SentencePieceProcessor()
        TextDecoder.sp1.Load(model_de)

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request.inputs["general_transformer_output"])).decode('utf-8')
        else:
            self.request_input = request_input["general_transformer_output"]
        self.istub = istub

    def Apply(self):
        self.decoded_tgt = self.sp1.DecodePieces(self.request_input.split(" "))

    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            tt = self.decoded_tgt.encode('utf-8')

            next_request = predict_pb2.PredictRequest()
            next_request.inputs['decoder_output'].CopyFrom(
              tf.make_tensor_proto(tt))

            return next_request
        else:
            result = dict()
            tt = self.decoded_tgt.encode('utf-8')
            result["decoder_output"] = tt
            return result