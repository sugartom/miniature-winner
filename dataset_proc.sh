BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
OUTPUT_DIR="${1:-wmt16_de_en}"
VOCAB_SIZE=32768
OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"




echo 'Shuffling'

shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.en > ${OUTPUT_DIR}/train.clean.en.shuffled
shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.de > ${OUTPUT_DIR}/train.clean.de.shuffled
cat ${OUTPUT_DIR}/train.clean.en.shuffled ${OUTPUT_DIR}/train.clean.de.shuffled > ${OUTPUT_DIR}/train.clean.en-de.shuffled.common

echo 'TOKENIZATION'
## Common
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en-de.shuffled.common \
  --model_prefix=${OUTPUT_DIR}/m_common --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.de.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.de.shuffled.BPE_common.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-de.src.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-de.ref.BPE_common.32K.tok

# Test Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE_common.32K.tok

## Language-dependent
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_en --vocab_size=${VOCAB_SIZE} --mode=train
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.de.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_de --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.de.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.de.shuffled.BPE.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-de.src.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-de.ref.BPE.32K.tok

# Test Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE.32K.tok
