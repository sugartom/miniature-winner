# A simple speech synthesis and speech recognition pipeline

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

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


# @contextmanager
# def timeit_context(name):
#     startTime = time.time()
#     yield
#     elapsedTime = time.time() - startTime
#     print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


# Initialize and setup all modules
taco = Tacotron_de()
taco.Setup()


# ============ Speech Recognition Modules ============
# deepspeech = Deepspeech2()
# deepspeech.Setup()

# jasper = Jasper()
# jasper.Setup()

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

# transformer_big = TransformerBig()
# transformer_big.Setup()

# conv_s2s = Convs2s()
# conv_s2s.Setup()

translation = transformer
# ============ Translation Modules ============

decoder = TextDecoder()
decoder.Setup()

# Input
# Input
input_audio, sr = librosa.load('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/inputs/226-131533-0000.wav')

wav = input_audio

print(wav.shape)

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

text = decoded_text.encode("utf-8")
# text = decoded_text

# Speech synthesis module
pre = taco.PreProcess([text])
# pre = taco.PreProcess(text)
app = taco.Apply(pre)
post = taco.PostProcess(*app)

print(post.shape)

wav = save_audio(post, "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/outputs", "unused", sampling_rate=16000, save_format="disk", n_fft=800)
# This part is out of the pipeline, just for debug purpose
