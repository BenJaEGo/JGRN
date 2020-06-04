from configs.dataset import *
from configs.parameters import *


class Settings(object):
    def __init__(self,
                 name,
                 feature_dir,
                 feature_prefix,
                 targets,
                 seeds,
                 n_epoch,
                 learning_rate,
                 lr_decay_rate,
                 salience
                 ):
        self.name = name
        self.feature_dir = feature_dir
        self.feature_prefix = feature_prefix
        self.targets = targets
        self.seeds = seeds
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.n_fold = 5
        self.salience = salience

    @property
    def dict(self):
        return self.__dict__


class Model(object):
    def __init__(self,
                 n_channel,
                 n_frequency,
                 n_in_channel,
                 n_filters_conv1,
                 n_filters_conv2,
                 n_units_fc1,
                 n_units_fc2,
                 n_units_attention,
                 n_poly_order,
                 n_poly_filter,
                 n_dims_one,
                 n_dims_two,
                 n_class,
                 batch_size,
                 l2_regularization,
                 cnn_act,
                 gnn_act,
                 ):
        self.n_channel = n_channel
        self.n_frequency = n_frequency
        self.n_in_channel = n_in_channel
        self.n_filters_c1 = n_filters_conv1
        self.n_filters_c2 = n_filters_conv2
        self.n_units_fc1 = n_units_fc1
        self.n_units_fc2 = n_units_fc2
        self.n_units_att = n_units_attention
        self.n_poly_order = n_poly_order
        self.n_poly_filter = n_poly_filter
        self.n_dims_one = n_dims_one
        self.n_dims_two = n_dims_two
        self.n_class = n_class
        self.batch_size = batch_size
        self.l2_regularization = l2_regularization
        self.cnn_act = cnn_act
        self.gnn_act = gnn_act

    @property
    def dict(self):
        return self.__dict__


class Config(object):
    def __init__(self,
                 setting,
                 model):
        self.setting = setting
        self.model = model


setting = Settings(name,
                   psd_feature_dir,
                   feature_prefix,
                   targets,
                   seeds,
                   n_epoch,
                   learning_rate,
                   lr_decay_rate,
                   salience)

model = Model(
    n_channel,
    n_frequency,
    n_in_channel,
    n_filters_conv1,
    n_filters_conv2,
    n_units_fc1,
    n_units_fc2,
    n_units_attention,
    n_poly_order,
    n_poly_filter,
    n_dims_one,
    n_dims_two,
    n_class,
    batch_size,
    l2_regularization,
    cnn_act,
    gnn_act,
)
config = Config(setting, model)
