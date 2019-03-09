import tensorflow as tf
import sys
from OpenSeq2Seq.open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
    create_logdir, create_model

def get_model(args, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(
            args, base_config, config_module, base_model, None)
    return model, checkpoint


class Jasper:

    def Setup(self):
        args_S2T = ["--config_file=OpenSeq2Seq/example_configs/speech2text/jasper_10x5_8gpus_dr_mp.py",
                    "--mode=interactive_infer",
                    "--logdir=checkpoints/jasper_10x5_dr_sync/checkpoint/",
                    "--batch_size_per_gpu=1",
                    ]
        self.model, checkpoint_S2T = get_model(args_S2T, "S2T")

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        vars_S2T = {}
        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            if "S2T" in v.name:
                vars_S2T["/".join(v.op.name.split("/")[1:])] = v

        saver_S2T = tf.train.Saver(vars_S2T)
        saver_S2T.restore(self.sess, checkpoint_S2T)

        self.fetches = [
            self.model.get_data_layer().input_tensors,
            self.model.get_output_tensors(),
        ]

    def PreProcess(self, input):
        feed_dict = self.model.get_data_layer().create_feed_dict(input)

        return feed_dict

    def Apply(self, feed_dict):
        inputs, outputs = self.sess.run(self.fetches, feed_dict=feed_dict)

        return inputs, outputs

    def PostProcess(self, inputs, outputs):
        results = self.model.infer(inputs, outputs)

        return results[0][0]
