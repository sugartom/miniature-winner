# A simple speech synthesis and speech recognition pipeline

import os
import pickle

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


simple_route_table = "Jasper-TextEncoder-Transformer"
measure_module = "Transformer"
route_table = simple_route_table

sess_id = "chain_audio-000"

pickle_directory = "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % measure_module
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)



for audioIndex in range(3):
  frame_id = 0
  duration_sum = 0.0
  count = 0
  while (frame_id < 100):
    start = time.time()

    # get input
    if (measure_module == "Wave2Letter" or measure_module == "Jasper"):
      # input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/vlc-record-2019-09-01-11h13m06s-226-131533-0000.wav-.wav')
      input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/outOne%s.mp3' % str(audioIndex).zfill(3))
      # input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/outTwo%s.mp3' % str(audioIndex).zfill(3))
      # input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/outFour%s.mp3' % str(audioIndex).zfill(3))
      wav = input_audio
      frame_info = "%s-%s" % (sess_id, frame_id)
      route_index = 0
      request_input = dict()
      request_input["client_input"] = wav
      request_input['frame_info'] = frame_info
      request_input['route_table'] = route_table
      request_input['route_index'] = route_index
    elif (measure_module == "TextEncoder"):
      # pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextEncoder"), str(frame_id).zfill(3))
      # pickle_input = "%s/audioOne%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextEncoder"), str(audioIndex).zfill(3), str(0).zfill(3))
      # pickle_input = "%s/audioTwo%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextEncoder"), str(audioIndex).zfill(3), str(0).zfill(3))
      pickle_input = "%s/audioFour%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextEncoder"), str(audioIndex).zfill(3), str(0).zfill(3))
      with open(pickle_input) as f:
        request_input = pickle.load(f)
    elif (measure_module == "Transformer" or measure_module == "TransformerBig"):
      # pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextEncoder", str(frame_id).zfill(3))
      # pickle_input = "%s/audioOne%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextEncoder", str(audioIndex).zfill(3), str(0).zfill(3))
      # pickle_input = "%s/audioTwo%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextEncoder", str(audioIndex).zfill(3), str(0).zfill(3))
      pickle_input = "%s/audioFour%s-%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextEncoder", str(audioIndex).zfill(3), str(0).zfill(3))
      with open(pickle_input) as f:
        request_input = pickle.load(f)
    # elif (measure_module == "TextDecoder"):
    #   pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/%s" % findPreviousModule(route_table, "TextDecoder"), str(frame_id).zfill(3))
    #   with open(pickle_input) as f:
    #     request_input = pickle.load(f)
    # elif (measure_module == "Tacotron_de"):
    #   pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/pickle_tmp/TextDecoder", str(frame_id).zfill(3))
    #   with open(pickle_input) as f:
    #     request_input = pickle.load(f)

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
    # elif (measure_module == "TextDecoder"):
    #   module_instance = TextDecoder()
    # elif (measure_module == "Tacotron_de"):
    #   module_instance = Tacotron_de()

    module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
    module_instance.Apply()
    next_request = module_instance.PostProcess(grpc_flag = False)

    next_request['frame_info'] = request_input['frame_info']
    next_request['route_table'] = request_input['route_table']
    next_request['route_index'] = request_input['route_index'] + 1

    end = time.time()
    # print("duration = %s" % (end - start))

    # if (measure_module == "Jasper"):
    #   print(next_request["speech_recognition_output"])
    # # if (measure_module == "Wave2Letter"):
    # # 	print(next_request["speech_recognition_output"])
    # elif (measure_module == "TextEncoder"):
    #   print(next_request["encoder_output"])
    # elif (measure_module == "Transformer"):
    #   print(next_request["FINAL"])
    # # elif (measure_module == "TextDecoder"):
    # #   print(next_request["decoder_output"])
    # # elif (measure_module == "Tacotron_de"):
    # # 	if (frame_id == 50):
  	 # #    save_audio(next_request["FINAL"], "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs", "unused", sampling_rate=16000, save_format="disk", n_fft=800)

    # # pickle_output = "%s/audioOne%s-%s" % (pickle_directory, str(audioIndex).zfill(3), str(frame_id).zfill(3))
    # # pickle_output = "%s/audioTwo%s-%s" % (pickle_directory, str(audioIndex).zfill(3), str(frame_id).zfill(3))
    # pickle_output = "%s/audioFour%s-%s" % (pickle_directory, str(audioIndex).zfill(3), str(frame_id).zfill(3))
    # with open(pickle_output, 'w') as f:
    #   pickle.dump(next_request, f)

    frame_id += 1

    if (frame_id > 10):
      duration_sum += (end - start)
      count += 1

    # time.sleep(1.0)

  print("average duration over %d run: %.3f sec" % (count, duration_sum / count))