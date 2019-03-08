import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import tensor_util

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
from open_seq2seq.models.text2speech import save_audio
import sentencepiece as spm


args_T2S = ["--config_file=example_configs/text2speech/tacotron_LJ_float.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/tacotron-LJ-float/checkpoint/",
            "--batch_size_per_gpu=1",
            ]


args_S2T = ["--config_file=example_configs/speech2text/ds2_large_8gpus_mp.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/deepspeech2/ds2_large/",
            "--batch_size_per_gpu=1",
            ]

args_T2T = ["--config_file=example_configs/text2text/en-de/transformer-bp-fp32.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/transformer-base/Transformer-FP32-H-256",
            "--batch_size_per_gpu=1",
            ]
# A simpler version of what run.py does. It returns the created model and
# its saved checkpoint


model_en = '/home/oscar/sdb3/data/wmt16_de_en/m_en.model'
model_de = '/home/oscar/sdb3/data/wmt16_de_en/m_de.model'

def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint

model_T2S, checkpoint_T2S = get_model(args_T2S, "T2S")
model_S2T, checkpoint_S2T = get_model(args_S2T, "S2T")
model_T2T, checkpoint_T2T = get_model(args_T2T, "T2T")

sp1 = spm.SentencePieceProcessor()
sp1.Load(model_en)
sp2 = spm.SentencePieceProcessor()
sp2.Load(model_de)

# line = "I was trained using Nvidia's Open Sequence to Sequence framework."

# Define the inference function
n_fft = 1024
sampling_rate = 22050


def get_t2t_interactive_infer_results(model, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    src_text = feed_dict[model.get_data_layer().input_tensors["source_tensors"][0]]
    src_text_length = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][1]].astype(np.int32)
    print(src_text)
    print(src_text_length.dtype)

    channel = grpc.insecure_channel('0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'text2text'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['src_text'].CopyFrom(
        tf.contrib.util.make_tensor_proto(src_text, shape=list(src_text.shape)))
    request.inputs['src_text_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(src_text_length, shape=list(src_text_length.shape)))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print('Result returned from rpc')

    tgt_txt = tensor_util.MakeNdarray(
        result_future.result().outputs['tgt_txt'])

    inputs = {"source_tensors" : [src_text, src_text_length]}
    outputs = [tgt_txt]

    return model.infer(inputs, outputs)

def get_t2s_interactive_infer_results(model, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    text = feed_dict[model.get_data_layer().input_tensors["source_tensors"][0]]
    text_length = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][1]]
    print('text shape: ', text.shape)
    print('text_length shape: ', text_length.shape)

    # inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    channel = grpc.insecure_channel('192.168.0.142:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'saved_models'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['text'].CopyFrom(
        tf.contrib.util.make_tensor_proto(text, shape=list(text.shape)))
    request.inputs['text_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(text_length, shape=list(text_length.shape)))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print('Result returned from rpc')

    audio_length = np.array(
        result_future.result().outputs['audio_length'].int_val)
    print("audio_length = ", audio_length)

    audio = tensor_util.MakeNdarray(
        result_future.result().outputs['audio_prediction'])
    print(audio)

    
    inputs = feed_dict
    outputs = [None, None, None, None, audio_length, audio]

    return model.infer(inputs, outputs)


def get_s2t_interactive_infer_results(model, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    audio = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][0]]
    audio_length = feed_dict[model.get_data_layer().input_tensors[
        "source_tensors"][1]]
    x_id = feed_dict[model.get_data_layer().input_tensors[
        "source_ids"][0]]

    print('audio shape: ', audio.shape)
    print('audio_length shape: ', audio_length.shape)

    # inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    channel = grpc.insecure_channel('0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'speech2text'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['audio'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio, shape=list(audio.shape)))
    request.inputs['audio_length'].CopyFrom(
        tf.contrib.util.make_tensor_proto(audio_length,
                                          shape=list(audio_length.shape)))
    request.inputs['x_id'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_id,
                                          shape=list(x_id.shape)))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print('Result returned from rpc')

    inputs = model.get_data_layer().input_tensors
    indices_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['indices_decoded_sequence'])
    values_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['values_decoded_sequence'])
    dense_shape_decoded_sequence = tensor_util.MakeNdarray(
        result_future.result().outputs['dense_shape_decoded_sequence'])

    outputs = tf.SparseTensorValue(indices=indices_decoded_sequence,
                                   values=values_decoded_sequence,
                                   dense_shape=dense_shape_decoded_sequence)

    outputs = [outputs]

    return model.infer(inputs, outputs)

def infer(line):
    print("Input English")
    print(line)

    # Generate speech
    results = get_t2s_interactive_infer_results(model_T2S, model_in=[line])
    audio_length = results[1][4][0]

    if model_T2S.get_data_layer()._both:
        prediction = results[1][5][0]

    else:
        prediction = results[1][1][0]

    prediction = prediction[:audio_length - 1, :]
    mag_prediction = model_T2S.get_data_layer().get_magnitude_spec(prediction)

    wav = save_audio(mag_prediction, "unused", "unused", sampling_rate=sampling_rate, save_format="np.array", n_fft=n_fft)
    wav = librosa.core.resample(wav, sampling_rate, 16000)
    print("Generated Audio")


    # Generate text
    results = get_s2t_interactive_infer_results(model_S2T, model_in=[wav])
    english_recognized = results[0][0]

    print("Recognized Speech")
    print(english_recognized)


    # Generate translation
    encoded_src_list = sp1.EncodeAsPieces(english_recognized)
    encoded_src = ' '.join([w for w in encoded_src_list])

    results = get_t2t_interactive_infer_results(model_T2T, model_in=[encoded_src])
    encoded_tgt = results[1][0]
    decoded_tgt = sp2.DecodePieces(encoded_tgt.split(" "))

    print("Translation")
    print(decoded_tgt)

while True:
    line = input()
    if line == "":
        break
    infer(line)

    print("Infer complete")
