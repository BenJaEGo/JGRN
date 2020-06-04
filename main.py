import os
import sys
import numpy as np
import tensorflow as tf

from configs.config import config
from utils.data import split_fns, get_samples

from models.model import Model
from utils.routines import run

from configs.parameters import CUDA_VISIBLE_DEVICES

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # 程序按需申请内存

for seed in config.setting.seeds:
    for target_idx in range(0, len(config.setting.targets)):
        target = config.setting.targets[target_idx]
        n_fold = config.setting.n_fold

        for fold_idx in np.arange(n_fold):

            logger = r"{}/{}/{}/{}/{}".format(
                "{}".format("checkpoints"),
                "{}".format(config.setting.feature_prefix),
                seed,
                target,
                fold_idx
            )

            if os.path.exists(logger):
                pass
            else:
                os.makedirs(logger)

            # std output to txt
            if config.setting.salience:
                sys.stdout = open(os.path.join(logger, "log.txt"), "w+")
            else:
                pass

            tr_samples_fns, te_samples_fns = split_fns(
                config.setting.feature_dir,
                target,
                seed
            )

            print("################################################")
            print("################ setting & parameters ##########")
            print("################################################")

            for key, value in config.setting.dict.items():
                print("{:<20} {}".format(key, value))

            for key, value in config.model.dict.items():
                print("{:<20} {}".format(key, value))

            n_input = config.model.n_channel * config.model.n_frequency

            prior = np.random.normal(
                loc=1, scale=0.0001,
                size=[n_input, n_input])

            print("################################################")
            print("################ loading feature fns ###########")
            print("################################################")

            print("################ training feature fns ##########")
            [print(fn) for fn in tr_samples_fns[fold_idx]]

            print("################ test feature fns ##############")
            [print(fn) for fn in te_samples_fns[fold_idx]]

            tr_samples, tr_labels = get_samples(tr_samples_fns[fold_idx])
            te_samples, te_labels = get_samples(te_samples_fns[fold_idx])

            print("training samples {}, training labels {}".format(tr_samples.shape, tr_labels.shape))
            print("test samples {}, test labels {}".format(te_samples.shape, te_labels.shape))

            tr_samples = tr_samples.reshape([-1, n_input])
            te_samples = te_samples.reshape([-1, n_input])

            tr_samples = np.log(tr_samples + 1e-6)
            te_samples = np.log(te_samples + 1e-6)

            _mean = np.mean(tr_samples, axis=0)
            _std = np.std(tr_samples, axis=0)

            tr_samples = (tr_samples - _mean) / (_std + 1e-6)
            te_samples = (te_samples - _mean) / (_std + 1e-6)

            tr_samples = np.expand_dims(tr_samples, -1)
            te_samples = np.expand_dims(te_samples, -1)

            print("################################################")
            print("################ stopping feature fns ##########")
            print("################################################")

            print("prior {}, training samples {}/{}, test samples {}/{}".format(
                prior.shape,
                tr_samples.shape, tr_labels.shape,
                te_samples.shape, te_labels.shape)
            )

            with tf.device("gpu: 0"):
                graph = tf.Graph()
                with graph.as_default():
                    model = Model(**config.model.dict)

                    with tf.Session(graph=graph, config=tf_config) as sess:
                        init = tf.global_variables_initializer()
                        sess.run(init)
                        run(sess, model,
                            config.setting.n_epoch, config.setting.learning_rate, config.setting.lr_decay_rate,
                            tr_samples, prior, tr_labels,
                            te_samples, prior, te_labels,
                            te_samples, prior, te_labels,
                            logger
                            )

            # std output to txt
            if config.setting.salience:
                sys.stdout.close()
            else:
                pass
