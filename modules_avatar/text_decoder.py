import sentencepiece as spm

class TextDecoder:

    def Setup(self):
        model_de = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_de.model'

        self.sp1 = spm.SentencePieceProcessor()
        self.sp1.Load(model_de)

    def Apply(self, input):
        decoded_tgt = self.sp1.DecodePieces(input.split(" "))
        return decoded_tgt
