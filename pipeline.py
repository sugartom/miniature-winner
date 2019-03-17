# A simple speech synthesis and speech recognition pipeline

from modules.Tacotron import Tacotron
from modules.Deepspeech2 import Deepspeech2
from modules.audio_resample import Resample
from modules.text_encoder import TextEncoder
from modules.Transformer import Transformer
from modules.text_decoder import TextDecoder
from modules.Jasper import Jasper
from modules.Wave2Letter import Wave2Letter
from modules.TransformerBig import TransformerBig
from modules.Convs2s import Convs2s

# Initialize and setup all modules
taco = Tacotron()
taco.Setup()


# ============ Speech Recognition Modules ============
deepspeech = Deepspeech2()
deepspeech.Setup()

jasper = Jasper()
jasper.Setup()

wave2letter = Wave2Letter()
wave2letter.Setup()

speech_recognition = wave2letter
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

translation = conv_s2s
# ============ Translation Modules ============

decoder = TextDecoder()
decoder.Setup()

# Input
text = "I was trained using Nvidia's Open Sequence to Sequence framework."

while(True):
	# Speech synthesis module
	pre = taco.PreProcess([text])
	app = taco.Apply(pre)
	post = taco.PostProcess(*app)

	# Resampling module
	wav = resample.Apply(post)

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

	# This part is out of the pipeline, just for debug purpose
	print("Translation")
	print(decoded_text)
