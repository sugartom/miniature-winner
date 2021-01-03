import threading
import cv2
import grpc
import time
import numpy as np
import os
import pickle
import librosa

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

import sys

sys.path.append('/home/yitao/Documents/edge/D2-system/')
from utils_d2 import misc

sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')
from modules_avatar.jasper_d2 import Jasper
from modules_avatar.text_encoder_d2 import TextEncoder
from modules_avatar.transformer_d2 import Transformer

ichannel = grpc.insecure_channel('0.0.0.0:8500')
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# Setup
Jasper.Setup()
TextEncoder.Setup()
Transformer.Setup()

# module_name = "jasper"
# module_name = "audio_encoder"
module_name = "transformer"

pickle_directory = "%s/pickle_d2/miniature-winner/%s" % (os.environ['RIM_DOCKER_SHARE'], module_name)
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)

batch_size = 1
parallel_level = 1
run_num = 10

input_audio, _ = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/vlc-record-2019-09-01-11h13m06s-226-131533-0000.wav-.wav')
client_input = input_audio

def runBatch(batch_size, run_num, tid):
  start = time.time()

  frame_id = 0

  while (frame_id < run_num):
    module_instance = misc.prepareModuleInstance(module_name)
    data_array = []

    if (module_name == "jasper"):
      request = dict()
      request["client_input"] = client_input
      data_dict = module_instance.GetDataDict(request, grpc_flag = False)
      data_array.append(data_dict)
    elif (module_name == "audio_encoder"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "miniature-winner", "jasper"), str(1).zfill(3))
      with open(pickle_input) as f:
        request = pickle.load(f)
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
    elif (module_name == "transformer"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "miniature-winner", "audio_encoder"), str(1).zfill(3))
      with open(pickle_input) as f:
        request = pickle.load(f)
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)

    batched_data_dict = module_instance.GetBatchedDataDict(data_array, batch_size)

    batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub)

    batched_result_array = module_instance.GetBatchedResultArray(batched_result_dict, batch_size)

    for i in range(len(batched_result_array)):
      # deal with the outputs of the ith input in the batch
      result_dict = batched_result_array[i]

      # each input might have more than one outputs
      result_list = module_instance.GetResultList(result_dict)

      for result in result_list:
        next_request = module_instance.GetNextRequest(result, grpc_flag = False)

        if (module_name == "jasper"):
          print(next_request["speech_recognition_output"])
        elif (module_name == "audio_encoder"):
          print(next_request["encoder_output"])
        elif (module_name == "transformer"):
          print(next_request["transformer_output"])

        # pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        # with open(pickle_output, 'w') as f:
        #   pickle.dump(next_request, f)

    frame_id += 1

  end = time.time()
  print("[Thread-%d] it takes %.3f sec to run %d batches of batch size %d" % (tid, end - start, run_num, batch_size))

# ========================================================================================================================

start = time.time()

thread_pool = []
for i in range(parallel_level):
  t = threading.Thread(target = runBatch, args = (batch_size, run_num, i))
  thread_pool.append(t)
  t.start()

for t in thread_pool:
  t.join()

end = time.time()
print("overall time = %.3f sec" % (end - start))
