# A simple speech synthesis and speech recognition pipeline

import os
import pickle
import threading

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

from modules_avatar.Wave2Letter_flexible import Wave2Letter
from modules_avatar.text_encoder_flexible import TextEncoder
from modules_avatar.Transformer_flexible import Transformer
from modules_avatar.text_decoder_flexible import TextDecoder
from modules_avatar.Tacotron_de_flexible import Tacotron_de
from modules_avatar.Jasper_flexible import Jasper
from modules_avatar.TransformerBig_flexible import TransformerBig

from modules_avatar.Deepspeech2 import Deepspeech2
from modules_avatar.audio_resample import Resample
# from modules_avatar.Jasper import Jasper
# from modules_avatar.TransformerBig import TransformerBig
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

def findPreviousModule(route_table, measure_module):
  tmp = route_table.split('-')
  for i in range(len(tmp)):
    if (tmp[i] == measure_module):
      return tmp[i - 1]

# {Wave2Letter, Jasper} - TextEncoder - {Transformer, TransformerBig}
simple_route_table = "Jasper-TextEncoder-Transformer"
measure_module = "Transformer"
route_table = simple_route_table

sess_id = "chain_audio-000"
frame_id = 0

input_fps = int(sys.argv[1])
total_frame = 120

def runFrame(measure_module, request_input, frame_id):
  if (measure_module == "Wave2Letter"):
    module_instance = Wave2Letter()
  elif (measure_module == "Jasper"):
    module_instance = Jasper()
  elif (measure_module == "TextEncoder"):
    module_instance = TextEncoder()
  elif (measure_module == "Transformer"):
    module_instance = Transformer()
  elif (measure_module == "TransformerBig"):
    module_instance = TransformerBig()

  module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  print("Finished frame %d for module %s" % (frame_id, measure_module))

  if (frame_id == 10):
    global stime
    stime = time.time()
  elif (frame_id == total_frame - 1):
    global etime
    etime = time.time()

if True:
# get input
  if (measure_module == "Wave2Letter" or measure_module == "Jasper"):
    input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/vlc-record-2019-09-01-11h13m06s-226-131533-0000.wav-.wav')
    wav = input_audio
    frame_info = "%s-%s" % (sess_id, "32")
    route_index = 0
    request_input = dict()
    request_input["client_input"] = wav
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  elif (measure_module == "TextEncoder"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextEncoder"), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "Transformer" or measure_module == "TransformerBig"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextEncoder", str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

print("input loaded")

while frame_id < total_frame:
  # frame_thread = threading.Thread(target = runFrame, args = (measure_module, sess_id, frame_id, reader,))
  frame_thread = threading.Thread(target = runFrame, args = (measure_module, request_input, frame_id,))
  frame_thread.start()

  time.sleep(1.0/input_fps)
  frame_id += 1

try:
  while True:
    time.sleep(60 * 60 * 24)
except KeyboardInterrupt:
  print("\nEnd by keyboard interrupt")
  print("<%f, %f> = %f over %d frames with fps of %f" % (float(stime), float(etime), float(etime) - float(stime), total_frame, (total_frame - 1 - 10) / (float(etime) - float(stime))))
