# A simple speech synthesis and speech recognition pipeline

from Tacotron import Tacotron
from Deepspeech2 import Deepspeech2
from audio_resample import Resample

# Initialize and setup all modules
taco = Tacotron()
taco.Setup()

deepspeech = Deepspeech2()
deepspeech.Setup()

resample = Resample()
resample.Setup()

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

# This part is out of the pipeline, just for debug purpose
english_recognized = post
print("Recognized Speech")
print(english_recognized)