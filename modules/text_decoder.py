import sentencepiece as spm

class TextEncoder:

    def Setup(self):
        model_de = '/home/oscar/sdb3/data/wmt16_de_en/m_de.model'

        self.sp1 = spm.SentencePieceProcessor()
        self.sp1.Load(model_de)

    def Apply(self, input):
        decoded_tgt = sp1.DecodePieces(input.split(" "))
        return decoded_tgt