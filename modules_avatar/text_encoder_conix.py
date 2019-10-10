import sentencepiece as spm

class TextEncoder:

    @staticmethod
    def Setup():
        model_en = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_en.model'

        TextEncoder.sp1 = spm.SentencePieceProcessor()
        TextEncoder.sp1.Load(model_en)

    def Apply(self, input):
        encoded_src_list = TextEncoder.sp1.EncodeAsPieces(input)
        encoded_src = ' '.join([w for w in encoded_src_list])
        return encoded_src
