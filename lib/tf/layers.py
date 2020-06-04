from lib.tf.ops import *


class ConvolutionLayer(object):
    def __init__(self,
                 name,
                 filter,
                 strides,
                 padding
                 ):
        # filter shape [filter_height, filter_width, in_channels, out_channels]
        self._name = name
        self._filter = filter
        self._strides = strides
        self._padding = padding

        self._weights = get_variable_he(
            name="{}/weights".format(self._name),
            shape=filter,
            trainable=True
        )

        self._biases = get_variable_constant(
            name="{}/biases".format(self._name),
            shape=[filter[-1]],
            trainable=True
        )

    def forward(self, input_tensor):
        out = tf.nn.bias_add(tf.nn.conv2d(input_tensor, self._weights, self._strides, self._padding), self._biases)
        return out


class ConvolutionTransposeLayer(object):
    def __init__(self,
                 name,
                 filter,
                 output_shape,
                 strides,
                 padding
                 ):
        # filter shape [height, width, output_channels, in_channels]
        self._name = name
        self._filter = filter
        self._output_shape = output_shape
        self._strides = strides
        self._padding = padding

        self._weights = get_variable_he(
            name="{}/weights".format(self._name),
            shape=filter,
            trainable=True
        )

        self._biases = get_variable_constant(
            name="{}/biases".format(self._name),
            shape=[filter[-2]],
            trainable=True
        )

    def forward(self, input_tensor):
        output_shape = [input_tensor.get_shape().as_list()[0]] + self._output_shape
        out = tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_tensor, self._weights, output_shape, self._strides, self._padding),
            self._biases)
        return out


class GraphConvolutionLayer(object):
    def __init__(self,
                 name,
                 nic,
                 npf,
                 npo):
        self._nic = nic
        self._npf = npf
        self._npo = npo
        self._name = name

        self._weights = get_variable_he(
            name="{}/weights".format(self._name),
            shape=[self._nic * self._npo, self._npf],
        )

        self._biases = get_variable_constant(
            name="{}/biases".format(self._name),
            shape=[self._npf],
            trainable=False
        )

    def forward(self, x, L):
        out = chebyshev5(x,
                         self._weights,
                         L,
                         self._npf,
                         self._npo)

        out = tf.nn.bias_add(out, self._biases)
        return out


class BatchNormLayer(object):
    """
    input shape [N, D]
    """

    def __init__(self,
                 name,
                 n_feature,
                 decay=0.99,
                 epsilon=1e-3,
                 trainable=True):
        self._decay = decay
        self._n_feature = n_feature
        self._name = name

        self._gamma = get_variable_constant(
            name="{}/gamma".format(self._name),
            shape=[self._n_feature],
            offset=1.,
            trainable=trainable
        )

        self._beta = get_variable_constant(
            name="{}/beta".format(self._name),
            shape=[self._n_feature],
            offset=0.,
            trainable=trainable
        )

        with tf.variable_scope(name):
            self._mean = tf.get_variable(name="mean",
                                         shape=[n_feature],
                                         initializer=tf.constant_initializer(
                                             value=0.
                                         ),
                                         trainable=False)

            self._variance = tf.get_variable(name="variance",
                                             shape=[n_feature],
                                             initializer=tf.constant_initializer(
                                                 value=1.
                                             ),
                                             trainable=False)

        self._epsilon = epsilon
        self._ema = tf.train.ExponentialMovingAverage(decay=decay)
        self._average_op = self._ema.apply([self._mean, self._variance])

    def normalize(self, x, is_training):
        def bn_train():

            if len(x.get_shape().as_list()) == 4:
                bn_dims = [0, 1, 2]
            elif len(x.get_shape().as_list()) == 3:
                bn_dims = [0, 1]
            elif len(x.get_shape().as_list()) == 2:
                bn_dims = [0]
            else:
                raise ValueError("Invalid tensor dims in bn layer.")

            mean, variance = tf.nn.moments(x, bn_dims)
            assign_mean = self._mean.assign(mean)
            assign_variance = self._variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(
                    x,
                    mean,
                    variance,
                    self._beta,
                    self._gamma,
                    self._epsilon,
                    name=None
                )

        def bn_test():
            mean = self._ema.average(self._mean)
            variance = self._ema.average(self._variance)
            local_beta = tf.identity(self._beta)
            local_gamma = tf.identity(self._gamma)
            return tf.nn.batch_normalization(
                x,
                mean,
                variance,
                local_beta,
                local_gamma,
                self._epsilon,
                name=None
            )

        out = tf.cond(is_training, lambda: bn_train(), lambda: bn_test())
        return out

    @property
    def average_op(self):
        return self._average_op


class PoolingOverFiltersLayer(object):
    def __init__(self,
                 name,
                 n_filters):
        self._n_filters = n_filters
        self._name = name

        self._weights = get_variable_he(
            name="{}/weights".format(self._name),
            shape=[1, self._n_filters, 1, 1],
        )

        self._biases = get_variable_constant(
            name="{}/biases".format(self._name),
            shape=[1],
            trainable=False
        )

    def forward(self, x):
        flow = tf.expand_dims(x, axis=-1)
        flow = tf.nn.conv2d(flow, self._weights, [1, 1, 1, 1], "VALID")
        flow = tf.nn.bias_add(flow, self._biases)
        out = tf.squeeze(flow, axis=[2, 3])
        return out


class DenseLayer(object):
    def __init__(self,
                 name,
                 n_input,
                 n_output,
                 trainable_biases=True):
        self._n_input = n_input
        self._n_output = n_output
        self._name = name

        self._weights = get_variable_he(
            name="{}/weights".format(self._name),
            shape=[self._n_input, self._n_output],
        )

        self._biases = get_variable_constant(
            name="{}/biases".format(self._name),
            shape=[self._n_output],
            trainable=trainable_biases
        )

    def forward(self, x):
        out = tf.nn.bias_add(tf.matmul(x, self._weights), self._biases)
        return out


class AttentionLayer(object):
    def __init__(self,
                 name,
                 n_input_units,
                 n_hidden_units,
                 n_output_units=1,
                 trainable_biases=True):
        self._name = name
        self._n_input_units = n_input_units
        self._n_hidden_units = n_hidden_units
        self._n_output_units = n_output_units
        self._trainable_biases = trainable_biases

        self._hidden_weights = get_variable_he(
            name="{}/hidden_weights".format(self._name),
            shape=[self._n_input_units, self._n_hidden_units]
        )
        self._output_weights = get_variable_he(
            name="{}/output_weights".format(self._name),
            shape=[self._n_hidden_units, self._n_output_units]
        )

        self._hidden_biases = get_variable_constant(
            name="{}/hidden_biases".format(self._name),
            shape=[self._n_hidden_units],
            trainable=self._trainable_biases
        )

        self._output_biases = get_variable_constant(
            name="{}/output_biases".format(self._name),
            shape=[self._n_output_units],
            trainable=self._trainable_biases
        )

    def forward(self, tensor):
        # input tensor [P, L], hidden weights [L, D], output weights [D, K]

        hidden = tf.nn.bias_add(tf.matmul(tensor, self._hidden_weights), self._hidden_biases)
        hidden = tanh(hidden)
        # [B, K=1]
        output = tf.nn.bias_add(tf.matmul(hidden, self._output_weights), self._output_biases)
        # [K=1, B]
        output = tf.transpose(output, [1, 0])

        attention = tf.nn.softmax(output, axis=1)
        return attention
