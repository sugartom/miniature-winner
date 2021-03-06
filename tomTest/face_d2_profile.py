import threading
import cv2
import grpc
import time
import numpy as np
import os
import pickle

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

sys.path.append('/home/yitao/Documents/edge/D2-system/')
from utils_d2 import misc
from modules_d2.video_reader import VideoReader

from modules_avatar.face_detector_d2_v2 import FaceDetector
from modules_avatar.prnet_image_cropper_d2_v2 import PRNetImageCropper
from modules_avatar.prnet_d2_v2 import PRNet

ichannel = grpc.insecure_channel('0.0.0.0:8500')
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

source_reader = VideoReader()
source_reader.Setup("/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/IMG_0003.mov")

# Setup
FaceDetector.Setup()
PRNetImageCropper.Setup()
PRNet.Setup()

# config
# module_name = "face_detector"
# module_name = "prnet_cropper"
module_name = "prnet_main"

# pickle_directory = "%s/pickle_d2/miniature-winner/%s" % (os.environ['RIM_DOCKER_SHARE'], module_name)
# if not os.path.exists(pickle_directory):
#   os.makedirs(pickle_directory)

batch_size = 1
parallel_level = 1
run_num = 6

sess_id = "chain_face-000"

client_input = misc.getClientInput("chain_face", source_reader)

def runBatch(batch_size, run_num, tid):
  start = time.time()

  frame_id = 1
  batch_id = 0

  while (batch_id < run_num):
    module_instance = misc.prepareModuleInstance(module_name)
    data_array = []

    if (module_name == "face_detector"):
      for i in range(batch_size):
        # client_input = misc.getClientInput("chain_face", source_reader)
        request = dict()
        request["client_input"] = client_input
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
        frame_id += 1
    elif (module_name == "prnet_cropper"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "miniature-winner", "face_detector"), str(1).zfill(3))
      with open(pickle_input) as f:
        request = pickle.load(f)
        request["output_flag"] = 1
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
      frame_id += 1
    elif (module_name == "prnet_main"):
      for i in range(batch_size):
        pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "miniature-winner", "prnet_cropper"), str(1).zfill(3))
        with open(pickle_input) as f:
          request = pickle.load(f)
          request["output_flag"] = 1
          data_dict = module_instance.GetDataDict(request, grpc_flag = False)
          data_array.append(data_dict)
        frame_id += 1
    
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

        if (module_name == "face_detector"):
          print(next_request["face_detector_output"])
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        # if (module_name == "prnet_cropper"):
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

    batch_id += 1

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
