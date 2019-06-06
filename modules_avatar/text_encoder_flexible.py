import sentencepiece as spm

import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2

class TextEncoder:

    @staticmethod
    def Setup():
        model_en = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_en.model'

        TextEncoder.sp1 = spm.SentencePieceProcessor()
        TextEncoder.sp1.Load(model_en)

    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["speech_recognition_output"])).decode('utf-8')
        else:
            self.request_input = request_input["speech_recognition_output"]
        self.istub = istub

    def Apply(self):
        encoded_src_list = self.sp1.EncodeAsPieces(self.request_input)
        self.encoded_src = ' '.join([w for w in encoded_src_list])
        
    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            tt = self.encoded_src.encode('utf-8')

            next_request = predict_pb2.PredictRequest()
            next_request.inputs['encoder_output'].CopyFrom(
              tf.make_tensor_proto(tt))

            return next_request
        else:
            result = dict()
            result["encoder_output"] = self.encoded_src
            return result