# A simple speech synthesis and speech recognition pipeline

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

from modules_avatar.Wave2Letter_flexible import Wave2Letter
from modules_avatar.text_encoder_flexible import TextEncoder
from modules_avatar.Transformer_flexible import Transformer
from modules_avatar.text_decoder_flexible import TextDecoder
from modules_avatar.Tacotron_de_flexible import Tacotron_de

from modules_avatar.Deepspeech2_flexible import Deepspeech2
from modules_avatar.Jasper_flexible import Jasper
from modules_avatar.TransformerBig_flexible import TransformerBig

from modules_avatar.audio_resample import Resample
from modules_avatar.Convs2s import Convs2s

from contextlib import contextmanager
import time
import librosa
from OpenSeq2Seq.open_seq2seq.models.text2speech import save_audio

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

ichannel = grpc.insecure_channel('0.0.0.0:8500')
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# Initialize and setup all modules
# ============ Speech Recognition Modules ============
deepspeech = Deepspeech2()
deepspeech.Setup()

jasper = Jasper()
jasper.Setup()

wave2letter = Wave2Letter()
wave2letter.Setup()

speech_recognition = wave2letter

# ============ Speech Recognition Modules ============
# resample = Resample()
# resample.Setup()

encoder = TextEncoder()
encoder.Setup()

# ============ Translation Modules ============
transformer = Transformer()
transformer.Setup()

transformer_big = TransformerBig()
transformer_big.Setup()

conv_s2s = Convs2s()
conv_s2s.Setup()

translation = transformer

# ============ Translation Modules ============
decoder = TextDecoder()
decoder.Setup()

# ============ Speech Synthesis Modules ============
taco = Tacotron_de()
taco.Setup()

simple_route_table = "Wave2Letter-TextEncoder-TransformerBig"
route_table = simple_route_table

sess_id = "chain_audio-000"
frame_id = 0

while (frame_id < 10):
  start = time.time()

  # input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/226-131533-0000.wav')
  input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/vlc-record-2019-09-01-11h13m06s-226-131533-0000.wav-.wav')
  wav = input_audio

  frame_info = "%s-%s" % (sess_id, frame_id)
  route_index = 0

  request_input = dict()
  request_input["client_input"] = wav
  request_input['frame_info'] = frame_info
  request_input['route_table'] = route_table
  request_input['route_index'] = route_index

  for i in range(len(route_table.split('-'))):
    current_model = route_table.split('-')[request_input['route_index']]

    if (current_model == "Wave2Letter"):
      module_instance = Wave2Letter()
    elif (current_model == "Deepspeech2"):
      module_instance = Deepspeech2()
    elif (current_model == "Jasper"):
      module_instance = Jasper()
    elif (current_model == "TextEncoder"):
      module_instance = TextEncoder()
    elif (current_model == "Transformer"):
      module_instance = Transformer()
    elif (current_model == "TransformerBig"):
      module_instance = TransformerBig()
    elif (current_model == "TextDecoder"):
      module_instance = TextDecoder()
    elif (current_model == "Tacotron_de"):
      module_instance = Tacotron_de()

    module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
    module_instance.Apply()
    next_request = module_instance.PostProcess(grpc_flag = False)

    next_request['frame_info'] = request_input['frame_info']
    next_request['route_table'] = request_input['route_table']
    next_request['route_index'] = request_input['route_index'] + 1

    request_input = next_request

    if (current_model == "Tacotron_de"):
      save_audio(request_input["FINAL"], "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs", "unused", sampling_rate=16000, save_format="disk", n_fft=800)

  end = time.time()
  print("duration = %s" % (end - start))

  frame_id += 1

