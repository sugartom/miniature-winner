# A simple speech synthesis and speech recognition pipeline

from modules.Tacotron import Tacotron
from modules.Deepspeech2 import Deepspeech2
from modules.audio_resample import Resample
from modules.text_encoder import TextEncoder
from modules.Transformer import Transformer
from modules.text_decoder import TextDecoder
from modules.Jasper import Jasper

# Initialize and setup all modules
taco = Tacotron()
taco.Setup()

#deepspeech = Deepspeech2()
#deepspeech.Setup()

jasper = Jasper()
jasper.Setup()
deepspeech = jasper

resample = Resample()
resample.Setup()

encoder = TextEncoder()
encoder.Setup()

transformer = Transformer()
transformer.Setup()

decoder = TextDecoder()
decoder.Setup()

# Input
text = "I was trained using Nvidia's Open Sequence to Sequence framework."

# Speech synthesis module
pre = taco.PreProcess([text])
app = taco.Apply(pre)
post = taco.PostProcess(*app)

# Resampling module
wav = resample.Apply(post)

# Speech recognition module
pre = deepspeech.PreProcess([wav])
app = deepspeech.Apply(pre)
post = deepspeech.PostProcess(*app)

# Encoding english text
encoded_text = encoder.Apply(post)

# Translation module
pre = transformer.PreProcess([encoded_text])
app = transformer.Apply(pre)
post = transformer.PostProcess(*app)

# Decoding German text
decoded_text = decoder.Apply(post)

# This part is out of the pipeline, just for debug purpose
print("Translation")
print(decoded_text)
