import tensorflow as tf
from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model

def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint


class Text2Speech:

    def Setup(self):
        args_T2S = ["--config_file=example_configs/text2speech/tacotron_LJ_float.py",
                    "--mode=interactive_infer",
                    "--logdir=/home/oscar/filesys/tacotron-LJ-float/checkpoint/",
                    "--batch_size_per_gpu=1",
                    ]
        self.model, checkpoint_T2S = get_model(args_T2S, "T2S")

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        vars_T2S = {}
        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            if "T2S" in v.name:
                vars_T2S["/".join(v.op.name.split("/")[1:])] = v

        saver_T2S = tf.train.Saver(vars_T2S)
        saver_T2S.restore(self.sess, checkpoint_T2S)

        self.fetches = [
            self.model.get_data_layer().input_tensors,
            self.model.get_output_tensors(),
        ]


    def PreProcess(self, input):
        feed_dict = self.model.get_data_layer().create_feed_dict(input)

        return feed_dict

    def Apply(self, feed_dict):

        inputs, outputs = self.sess.run(self.fetches, feed_dict=input)

        return inputs, outputs

    def PostProcess(self, inputs, outputs):
        result = self.model.infer(inputs, outputs)

        audio_length = results[1][4][0]

        if model_T2S.get_data_layer()._both:
            prediction = results[1][5][0]

        else:
            prediction = results[1][1][0]

        prediction = prediction[:audio_length - 1, :]
        mag_prediction = self.model.get_data_layer().get_magnitude_spec(prediction)

        return mag_prediction