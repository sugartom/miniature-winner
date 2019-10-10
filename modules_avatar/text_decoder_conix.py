import sentencepiece as spm

class TextDecoder:

    @staticmethod
    def Setup():
        model_de = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/data/translation_data/m_de.model'

        TextDecoder.sp1 = spm.SentencePieceProcessor()
        TextDecoder.sp1.Load(model_de)

    def Apply(self, input):
        decoded_tgt = TextDecoder.sp1.DecodePieces(input.split(" "))
        return decoded_tgt
