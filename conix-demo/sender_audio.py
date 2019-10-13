import time
import io
import threading
from Queue import Queue
import wave

import soundfile as sf
import pyaudio

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import grpc


def get_wav_data(raw_data):
  # generate the WAV file contents
  with io.BytesIO() as wav_file:
    wav_writer = wave.open(wav_file, "wb")
    try:  # note that we can't use context manager, since that was only added in Python 3.4
      wav_writer.setframerate(RATE)
      wav_writer.setsampwidth(WIDTH)
      wav_writer.setnchannels(CHANNELS)
      wav_writer.writeframes(raw_data)
      # wav_data = wav_file.getvalue()
      wav_file.seek(0)
      data, samplerate = sf.read(wav_file)
    finally:  # make sure resources are cleaned up
        wav_writer.close()
  return data, samplerate

WIDTH = 2
CHANNELS = 1
RATE = 22050
CHUNK = 4096
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 5

# Set up visualizer channel
visualize_channel = grpc.insecure_channel("192.168.1.9:50051")
visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)

worker_channel = grpc.insecure_channel("192.168.1.9:50101")
worker_stub = prediction_service_pb2_grpc.PredictionServiceStub(worker_channel)

# Setup pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# A worker that does the inference, this is the actual pipeline
q = Queue()
def worker():
  while True:
    wav = q.get()

    audio_request = predict_pb2.PredictRequest()
    audio_request.inputs['input_audio'].CopyFrom(
      tf.make_tensor_proto(wav))

    audio_result = worker_stub.Predict(audio_request, 10.0)

    q.task_done()

t = threading.Thread(target=worker)
t.daemon = True
t.start()

# A while loop that fetches audio
frames = []
chunk_count = 0

while True:
  data = stream.read(CHUNK)

  # Send the audio immediately so that we don't delay the playback
  request = predict_pb2.PredictRequest()
  request.inputs['audio'].CopyFrom(
    tf.contrib.util.make_tensor_proto(data))
  result = visualize_stub.Predict(request, 10.0)

  # Only do inference when we have RECORD_SECONDS of audio
  frames.append(data)
  chunk_count += 1
  if chunk_count > int(RATE / CHUNK * RECORD_SECONDS):
    print("send to inference")
    # convert into wav 
    raw_data = b''.join(frames)
    wav, sr = get_wav_data(raw_data)

    # Put into a queue to do inference, this one should be async
    q.put(wav)
    frames = []
    chunk_count = 0


stream.stop_stream()
stream.close()

p.terminate()