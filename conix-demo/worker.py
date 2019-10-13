from concurrent import futures
import time
import threading
from Queue import Queue

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

from modules_avatar.prnet_conix import PRNet
from modules_avatar.prnet_image_cropper import PRNetImageCropper
from modules_avatar.face_detector import FaceDetector

from modules_avatar.Deepspeech2_conix import Deepspeech2
from modules_avatar.text_encoder_conix import TextEncoder
from modules_avatar.Transformer_conix import Transformer
from modules_avatar.text_decoder_conix import TextDecoder

class RimWorker(prediction_service_pb2_grpc.PredictionServiceServicer):
  def __init__(self):
    PRNet.Setup()
    FaceDetector.Setup()
    PRNetImageCropper.Setup()

    Deepspeech2.Setup()
    TextEncoder.Setup()
    Transformer.Setup()
    TextDecoder.Setup()

    channel = grpc.insecure_channel('localhost:8500')
    self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    visualize_channel = grpc.insecure_channel("192.168.1.9:50051")
    self.visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)

    self.q = Queue()

    t = threading.Thread(target = self.SendToVisualizer)
    t.daemon = True
    t.start()

  def SendToVisualizer(self):
    while True:
      item = self.q.get()
      result = self.visualize_stub.Predict(item, 10.0)
      self.q.task_done()

  def Predict(self, request, context):
    if ("input_image" in request.inputs):
      print("[%.6f] Received face request" % time.time())

      image = tensor_util.MakeNdarray(request.inputs["input_image"])

      worker_request = predict_pb2.PredictRequest()
      worker_request.inputs['input_image'].CopyFrom(
        tf.make_tensor_proto(image))

      face_detector = FaceDetector()
      face_detector.PreProcess(worker_request, self.stub)
      face_detector.Apply()
      prnet_request = face_detector.PostProcess()

      if "bounding_box" in prnet_request.inputs:
        prnet_image_cropper = PRNetImageCropper()
        prnet_image_cropper.PreProcess(prnet_request, self.stub)
        prnet_image_cropper.Apply()
        next_request = prnet_image_cropper.PostProcess()

        prn = PRNet()
        prn.PreProcess(next_request, self.stub)
        prn.Apply()
        key_points, vertices = prn.PostProcess()

        visualizer_request = predict_pb2.PredictRequest()
        visualizer_request.inputs['prnet_output'].CopyFrom(
          tf.make_tensor_proto(key_points))
        visualizer_request.inputs['vertices'].CopyFrom(
          tf.make_tensor_proto(vertices))
        self.q.put(visualizer_request)

      worker_response = predict_pb2.PredictResponse()
      worker_response.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))

      return worker_response

    elif ("input_audio" in request.inputs):
      print("[%.6f] Received audio request" % time.time())

      wav = tensor_util.MakeNdarray(request.inputs["input_audio"])

      # Speech recognition module
      speech_recognition = Deepspeech2()
      pre = speech_recognition.PreProcess([wav], self.stub)
      app = speech_recognition.Apply(pre)
      post = speech_recognition.PostProcess(*app)

      print("post: %s" % post)

      # Encoding english text
      encoder = TextEncoder()
      encoded_text = encoder.Apply(post)

      # Translation module
      translation = Transformer()
      pre = translation.PreProcess([encoded_text], self.stub)
      app = translation.Apply(pre)
      post = translation.PostProcess(*app)

      # Decoding German text
      decoder = TextDecoder()
      decoded_text = decoder.Apply(post)

      print("Translation: %s" % decoded_text)

      visualizer_request = predict_pb2.PredictRequest()
      visualizer_request.inputs['subtitle'].CopyFrom(
        tf.contrib.util.make_tensor_proto(decoded_text))
      self.q.put(visualizer_request)

      worker_response = predict_pb2.PredictResponse()
      worker_response.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))

      return worker_response

def main(_):
  _ONE_DAY_IN_SECONDS = 60 * 60 * 24
  MAX_MESSAGE_LENGTH = 1024 * 1024 * 256
  MAX_WORKERS = 600

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(RimWorker(), server)
  server.add_insecure_port("192.168.1.9:50101")
  server.start()

  print("started worker's stub\n")

  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  tf.app.run()
