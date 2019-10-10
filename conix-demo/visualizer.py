from concurrent import futures
import time
import math
import logging

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

import cv2
from PRNet.utils.cv_plot import plot_kpt, plot_vertices
import pymesh
import threading
from Queue import Queue
from tensorflow.python.framework import tensor_util
import numpy as np


import pyaudio

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=22500,
                output=True)


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,350)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


subtitles = Queue()

q = Queue()
def worker():
    display_subtitle = ""
    while True:
        item = q.get()
        image = np.zeros((480, 640))
        if item is not None:
            vertices = item
            show_img = plot_vertices(np.zeros_like(image), vertices)
        else:
            show_img = image 
                # Display the resulting frame

        if not subtitles.empty():
            text = subtitles.get()
            subtitles.task_done()
            display_subtitle = text
        cv2.putText(show_img,display_subtitle, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imshow('frame',show_img)


        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        q.task_done()




class FakeServer(prediction_service_pb2_grpc.PredictionServiceServicer):
  def Predict(self, request,     context):
    """Predict -- provides access to loaded TensorFlow model.
    """
    global q
    global stream
    if "vertices" in request.inputs:
        print("vertices")
        vertices = tensor_util.MakeNdarray(request.inputs["vertices"])
        q.put(vertices)
    elif "audio" in request.inputs:
        print('audio')
        # audio = tensor_util.MakeNdarray(request.inputs['audio'])
        print(type(request.inputs['audio'].string_val[0]))
        audio = request.inputs['audio'].string_val[0]
        stream.write(audio)
    elif "subtitle" in request.inputs:
        print('subtitle')
        subtitles.put(request.inputs['subtitle'].string_val[0])




    dumbresult = predict_pb2.PredictResponse()
    dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
    return dumbresult



def serve():

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
        FakeServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    # server.wait_for_termination()


    stream.stop_stream()
    stream.close()

    p.terminate()
    q.join()       # block until all tasks are donet
    subtitles.join()

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24

    try:
      while True:
        time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
      server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()

