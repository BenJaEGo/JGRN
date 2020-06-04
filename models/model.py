from modules.module import *
import tensorflow as tf


class Model(object):
    def __init__(self,

                 n_channel,
                 n_frequency,

                 n_in_channel,

                 n_filters_c1,
                 n_filters_c2,
                 n_units_fc1,
                 n_units_fc2,
                 n_units_att,

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

        self._cnn_act = cnn_act
        self._gnn_act = gnn_act

        self._n_channel = n_channel
        self._n_frequency = n_frequency
        self._n_input = self._n_frequency * self._n_channel

        self._n_in_channel = n_in_channel
        self._n_class = n_class

        self._n_filters_conv1 = n_filters_c1
        self._n_filters_conv2 = n_filters_c2
        self._n_units_fc1 = n_units_fc1
        self._n_units_fc2 = n_units_fc2
        self._n_units_attention = n_units_att

        self._n_poly_order = n_poly_order
        self._n_poly_filter = n_poly_filter
        self._n_dims_one = n_dims_one
        self._n_dims_two = n_dims_two

        self._n_frequency = n_frequency

        self._batch_size = batch_size

        self._l2_regularization = l2_regularization

        with tf.name_scope('INPUT'):
            self._ph_prior = tf.placeholder(tf.float32,
                                            [
                                                self._n_input,
                                                self._n_input
                                            ],
                                            'ph_prior')

            self._ph_sample = tf.placeholder(tf.float32,
                                             [self._batch_size,
                                              self._n_input,
                                              self._n_in_channel],
                                             'ph_sample')

            self._ph_bn = tf.placeholder(tf.bool, [], 'ph_bn')
            self._ph_label = tf.placeholder(tf.int32, [self._batch_size], 'ph_label')
            self._ph_lr = tf.placeholder(tf.float32, [], 'ph_lr')

        with tf.name_scope("BUILD"):

            cnn_module = ConvolutionModule(
                "CNN",
                H=n_frequency, W=n_frequency, n_channel=n_channel, n_frequency=n_frequency,
                n_filters_conv1=self._n_filters_conv1,
                n_filters_conv2=self._n_filters_conv2,
                n_units_fc1=self._n_units_fc1,
                n_units_fc2=self._n_units_fc2,
                n_units_attention=self._n_units_attention
            )

            gnn_module = GraphConvolutionModule(
                "GNN",
                n_input=self._n_input,
                n_dims_one=n_dims_one,
                n_dims_two=n_dims_two,
                nic=self._n_in_channel,
                npf=self._n_poly_filter,
                npo=self._n_poly_order,

            )

            self._initial_graph = tf.linalg.set_diag(self._ph_prior, diagonal=tf.zeros(self._ph_prior.shape[0:-1]))

            with tf.name_scope("ATTENTION"):
                self._dense_graph = cnn_module.forward(
                    self._initial_graph, self._ph_bn, self._cnn_act)

            print("produced image shape : {}".format(self._dense_graph.get_shape().as_list()))

            with tf.name_scope("GRAPH"):
                self._sparse_graph = self._dense_graph
                self._sparse_graph_mean = tf.reduce_mean(self._sparse_graph, axis=[0])
                self._sparse_graph = self._sparse_graph - self._sparse_graph_mean

                self._sparse_graph = tanh(self._sparse_graph)
                self._sparse_graph = relu(self._sparse_graph)

                self._sparse_graph = tf.linalg.set_diag(self._sparse_graph,
                                                        diagonal=tf.zeros(self._dense_graph.shape[0:-1]))

                self._sparse_graph = (self._sparse_graph + tf.transpose(self._sparse_graph)) / 2

                self._L = laplacian(self._sparse_graph)

                self._logits, self._features = gnn_module.forward(self._ph_sample, self._L, self._ph_bn, self._gnn_act)

        with tf.name_scope("STATS"):
            self._op_prediction = tf.argmax(self._logits, axis=1, name="predictions")
            self._op_probability = tf.nn.softmax(self._logits, axis=1)

        with tf.name_scope("OPTIMIZER"):
            self._global_step = tf.Variable(tf.convert_to_tensor(0.), name="global_step", trainable=False)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._ph_lr, name="optimizer")

        with tf.name_scope("LOSS"):
            self._t_vars = tf.trainable_variables()
            print("#############################################")
            print("############### parameters ##################")
            print("#############################################")

            print("#############################################")
            print("trainable variables: ")
            [print("{}, {}".format(var.name, var.get_shape().as_list())) for var in self._t_vars]

            self._clf_loss = tf.reduce_mean(
                tf.losses.hinge_loss(
                    labels=tf.one_hot(self._ph_label, depth=n_class),
                    logits=self._logits))

            self._l2_vars = [var for var in self._t_vars if "weights" in var.name]

            print("#############################################")
            print("L2 normalization variables: ")
            [print(var.name) for var in self._l2_vars]
            print("#############################################")

            with tf.name_scope("L2NORM"):
                if self._l2_regularization is not None:
                    self._l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self._l2_vars]) * self._l2_regularization
                else:
                    self._l2_loss = tf.convert_to_tensor(0.)

            self._loss = self._clf_loss + self._l2_loss

        with tf.name_scope('AVERAGES'):
            averages = tf.train.ExponentialMovingAverage(0.9)
            averages_op = averages.apply(
                [self._loss,
                 self._clf_loss,
                 self._l2_loss,
                 ])
            with tf.control_dependencies([averages_op]):
                self._loss_average = tf.identity(averages.average(self._loss), name='loss_average')
                self._l2_loss_average = tf.identity(averages.average(self._l2_loss), name='l2_loss_average')
                self._clf_loss_average = tf.identity(averages.average(self._clf_loss), name='ce_loss_average')

        with tf.name_scope("TRAINING"):
            self._saver = tf.train.Saver(max_to_keep=1)

            bn_update_ops = cnn_module.bn_update_ops + gnn_module.bn_update_ops

            with tf.control_dependencies(bn_update_ops):
                self._grads = tf.gradients(ys=self._loss, xs=self._t_vars)
                self._op_train = self._optimizer.apply_gradients(
                    grads_and_vars=zip(self._grads, self._t_vars),
                    global_step=self._global_step,
                    name="train_op")

            self._saver = tf.train.Saver(max_to_keep=1)

    @property
    def ph_sample(self):
        return self._ph_sample

    @property
    def ph_priors(self):
        return self._ph_prior

    @property
    def ph_label(self):
        return self._ph_label

    @property
    def ph_bn(self):
        return self._ph_bn

    @property
    def n_channel(self):
        return self._n_channel

    @property
    def ph_lr(self):
        return self._ph_lr

    @property
    def loss(self):
        return self._loss

    @property
    def op_train(self):
        return self._op_train

    @property
    def loss_average(self):
        return self._loss_average

    @property
    def learning_rate(self):
        return self._ph_lr

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def op_prediction(self):
        return self._op_prediction

    @property
    def op_probability(self):
        return self._op_probability

    @property
    def op_saver(self):
        return self._saver

    @property
    def L(self):
        return self._L

    @property
    def ce_loss_average(self):
        return self._clf_loss_average

    @property
    def ce_loss(self):
        return self._clf_loss

    @property
    def l2_loss_average(self):
        return self._l2_loss_average

    @property
    def features(self):
        return self._features


if __name__ == "__main__":
    model = Model(
        n_channel=6,

        n_frequency=114,

        n_in_channel=1,

        n_filters_c1=8,
        n_filters_c2=16,
        n_units_fc1=256,
        n_units_fc2=128,
        n_units_att=128,

        n_poly_order=1,
        n_poly_filter=64,
        n_dims_one=256,
        n_dims_two=128,

        n_class=2,

        batch_size=64,
        l2_regularization=1e-4,

        cnn_act=tf.nn.relu,
        gnn_act=tf.nn.tanh,
    )
