import sentencepiece as spm

class TextEncoder:

    def Setup(self):
        model_en = 'data/translation_data/m_en.model'

        self.sp1 = spm.SentencePieceProcessor()
        self.sp1.Load(model_en)

    def Apply(self, input):
        encoded_src_list = self.sp1.EncodeAsPieces(input)
        encoded_src = ' '.join([w for w in encoded_src_list])
        return encoded_src
