# A simple speech synthesis and speech recognition pipeline

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')

from modules_avatar.Tacotron_de import Tacotron_de
from modules_avatar.Deepspeech2 import Deepspeech2
from modules_avatar.audio_resample import Resample
from modules_avatar.text_encoder import TextEncoder
from modules_avatar.Transformer import Transformer
from modules_avatar.text_decoder import TextDecoder
from modules_avatar.Jasper import Jasper
from modules_avatar.Wave2Letter import Wave2Letter
from modules_avatar.TransformerBig import TransformerBig
from modules_avatar.Convs2s import Convs2s

from contextlib import contextmanager
import time
import librosa
from OpenSeq2Seq.open_seq2seq.models.text2speech import save_audio

import soundfile as sf
import pyaudio
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import grpc

import io


import threading
from Queue import Queue
import wave

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
visualize_channel = grpc.insecure_channel("localhost:50051")
visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)


# Setup pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)



# ============ Speech Recognition Modules ============
deepspeech = Deepspeech2()
deepspeech.Setup()

jasper = Jasper()
jasper.Setup()

wave2letter = Wave2Letter()
wave2letter.Setup()

speech_recognition = deepspeech
# ============ Speech Recognition Modules ============

resample = Resample()
resample.Setup()

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




# A worker that does the inference, this is the actual pipeline
q = Queue()
def worker():
    while True:
    	wav = q.get()

        # Speech recognition module
        pre = speech_recognition.PreProcess([wav])
        app = speech_recognition.Apply(pre)
        post = speech_recognition.PostProcess(*app)

        print(post)

        # Encoding english text
        encoded_text = encoder.Apply(post)

        # Translation module
        pre = translation.PreProcess([encoded_text])
        app = translation.Apply(pre)
        post = translation.PostProcess(*app)

        # Decoding German text
        decoded_text = decoder.Apply(post)

        print("Translation")
        print(decoded_text)


        request = predict_pb2.PredictRequest()
        request.inputs['subtitle'].CopyFrom(
            tf.contrib.util.make_tensor_proto(decoded_text))

        print(request.inputs['subtitle'])
        result = visualize_stub.Predict(request, 10.0)

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