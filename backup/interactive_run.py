import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
import matplotlib.pyplot as plt

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model
from open_seq2seq.models.text2speech import save_audio

args_T2S = ["--config_file=example_configs/text2speech/tacotron_LJ_float.py",
            "--mode=interactive_infer",
            "--logdir=/home/oscar/filesys/tacotron-LJ-float/checkpoint/",
            "--batch_size_per_gpu=1",
            ]

# A simpler version of what run.py does. It returns the created model and
# its saved checkpoint


def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint

model_T2S, checkpoint_T2S = get_model(args_T2S, "T2S")

# Create the session and load the checkpoints
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
vars_T2S = {}
for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    if "T2S" in v.name:
        vars_T2S["/".join(v.op.name.split("/")[1:])] = v
saver_T2S = tf.train.Saver(vars_T2S)
saver_T2S.restore(sess, checkpoint_T2S)

# line = "I was trained using Nvidia's Open Sequence to Sequence framework."

# Define the inference function
n_fft = model_T2S.get_data_layer().n_fft
sampling_rate = model_T2S.get_data_layer().sampling_rate


def get_interactive_infer_results(model, sess, model_in):
    fetches = [
        model.get_data_layer().input_tensors,
        model.get_output_tensors(),
    ]

    feed_dict = model.get_data_layer().create_feed_dict(model_in)

    inputs, outputs = sess.run(fetches, feed_dict=feed_dict)

    # export_path = "/tmp/saved_models/4"
    # print('Exporting trained model to', export_path)
    # print('len(model.get_output_tensors()) = ',
    #       len(model.get_output_tensors()))

    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # # Define input tensors
    # text = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][0])
    # text_length = tf.saved_model.utils.build_tensor_info(
    #     model.get_data_layer().input_tensors["source_tensors"][1])

    # # Define output tensors
    # predicted_final_specs = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[1])
    # attention_mask = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[2])
    # stop_token_pred = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[3])
    # audio_length = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[4])
    # audio_prediction = tf.saved_model.utils.build_tensor_info(
    #     model.get_output_tensors()[5])

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'text': text, 'text_length': text_length},
    #         outputs={'predicted_final_specs': predicted_final_specs,
    #                  'attention_mask': attention_mask,
    #                  'stop_token_pred': stop_token_pred,
    #                  'audio_length': audio_length,
    #                  'audio_prediction': audio_prediction},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'predict_output':
    #             prediction_signature,
    #     },
    #     main_op=tf.tables_initializer(),
    #     strip_default_attrs=True)

    # builder.save()

    return model.infer(inputs, outputs)


def infer(line):
    print("Input English")
    print(line)

    # Generate speech
    results = get_interactive_infer_results(model_T2S, sess, model_in=[line])
    audio_length = results[1][4][0]

    if model_T2S.get_data_layer()._both:
        prediction = results[1][5][0]

    else:
        prediction = results[1][1][0]

    prediction = prediction[:audio_length - 1, :]
    mag_prediction = model_T2S.get_data_layer().get_magnitude_spec(prediction)

    mag_prediction_squared = np.clip(mag_prediction, a_min=0, a_max=255)
    mag_prediction_squared = mag_prediction_squared**1.5
    mag_prediction_squared = np.square(mag_prediction_squared)

    mel_basis = librosa.filters.mel(
        sr=22050, n_fft=1024, n_mels=80, htk=True, norm=None)
    mel = np.dot(mel_basis, mag_prediction_squared.T)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    np.save("spec2", mel)

    wav = save_audio(mag_prediction,
                     "/home/oscar/filesys/tacotron-LJ-float/checkpoint/", "1",
                     sampling_rate=sampling_rate, save_format="disk",
                     n_fft=n_fft)
    print("Generated Audio")


while True:
    line = input()
    if line == "":
        break
    infer(line)

    print("Infer complete")
