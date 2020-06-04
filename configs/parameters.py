import tensorflow as tf

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

CUDA_VISIBLE_DEVICES = "0"

targets = ["pat003", "pat004", "pat005", "pat009", "pat010", "pat017", "pat018", "pat020", "pat021"]

salience = True

n_epoch = 15

n_in_channel = 1

n_filters_conv1 = 8
n_filters_conv2 = 16
n_units_fc1 = 256
n_units_fc2 = 128
n_units_attention = 128

n_poly_order = 1
n_poly_filter = 16
n_dims_one = 256
n_dims_two = 128

batch_size = 64
learning_rate = 1e-4
lr_decay_rate = 0.9

l2_regularization = 1e-4

cnn_act = tf.nn.relu
gnn_act = tf.nn.tanh
