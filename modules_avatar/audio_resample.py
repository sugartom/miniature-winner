import librosa

from OpenSeq2Seq.open_seq2seq.models.text2speech import save_audio


class Resample:
	def Setup(self):
		self.n_fft = 1024
		self.sampling_rate = 22050
		

	def Apply(self, input):
		wav = save_audio(input, "unused", "unused", sampling_rate=self.sampling_rate, save_format="np.array", n_fft=self.n_fft)
		wav = librosa.core.resample(wav, self.sampling_rate, 16000)
		return wav
