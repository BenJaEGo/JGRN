from lib.tf.layers import *
import math


class ConvolutionModule(object):
    def __init__(self,
                 name,
                 H, W, n_channel, n_frequency,
                 n_filters_conv1,
                 n_filters_conv2,
                 n_units_fc1,
                 n_units_fc2,
                 n_units_attention):
        self._name = name

        self._n_channel = n_channel
        self._n_frequency = n_frequency

        self._n_filters_conv1 = n_filters_conv1
        self._n_filters_conv2 = n_filters_conv2

        self._n_units_fc1 = n_units_fc1
        self._n_units_fc2 = n_units_fc2

        self._n_units_attention = n_units_attention

        self.H = H
        self.W = W

        self.K1 = 4
        self.S1 = 3

        self.K2 = 4
        self.S2 = 3

        # padding convolution, "SAME" padding
        self.H1 = math.ceil(self.H / self.S1)
        self.W1 = math.ceil(self.W / self.S1)

        self.H2 = math.ceil(self.H1 / self.S2)
        self.W2 = math.ceil(self.W1 / self.S2)

        self._conv1 = ConvolutionLayer(
            name="{}/conv1".format(self._name),
            filter=[self.K1, self.K1, 1, self._n_filters_conv1],
            strides=[1, self.S1, self.S1, 1],
            padding="SAME"
        )

        self._conv1BN = BatchNormLayer("{}/conv1BN".format(name),
                                       self._n_filters_conv1,
                                       trainable=True)

        self._conv2 = ConvolutionLayer(
            name="{}/conv2".format(self._name),
            filter=[self.K2, self.K2, self._n_filters_conv1, self._n_filters_conv2],
            strides=[1, self.S2, self.S2, 1],
            padding="SAME"
        )

        self._conv2BN = BatchNormLayer("{}/conv2BN".format(name),
                                       self._n_filters_conv2,
                                       trainable=True)

        self._deconv1 = ConvolutionTransposeLayer(
            name="{}/deconv1".format(self._name),
            filter=[self.K2, self.K2, self._n_filters_conv1, self._n_filters_conv2],
            output_shape=[self.H1, self.W1, self._n_filters_conv1],
            strides=[1, self.S2, self.S2, 1],
            padding="SAME"
        )

        self._dconv1BN = BatchNormLayer("{}/deconv1BN".format(name),
                                        self._n_filters_conv1,
                                        trainable=True)

        self._deconv2 = ConvolutionTransposeLayer(
            name="{}/deconv2".format(self._name),
            filter=[self.K1, self.K1, 1, self._n_filters_conv1],
            output_shape=[self.H, self.W, 1],
            strides=[1, self.S1, self.S1, 1],
            padding="SAME"
        )

        self._dense1 = DenseLayer(
            name="{}/dense1".format(self._name),
            n_input=self.H2 * self.W2 * self._n_filters_conv2,
            n_output=self._n_units_fc1,
            trainable_biases=True
        )

        self._dense1BN = BatchNormLayer("{}/dense1BN".format(name),
                                        self._n_units_fc1,
                                        trainable=True)

        self.dense2 = DenseLayer(
            name="{}/dense2".format(self._name),
            n_input=self._n_units_fc1,
            n_output=self._n_units_fc2,
            trainable_biases=True
        )

        self._dense2BN = BatchNormLayer("{}/dense2BN".format(name),
                                        self._n_units_fc2,
                                        trainable=True)

        self._attention = AttentionLayer(
            name="{}/attention".format(self._name),
            n_input_units=self._n_units_fc2,
            n_hidden_units=self._n_units_attention,
            n_output_units=1,
            trainable_biases=True
        )

        self.bn_update_ops = [
            self._dconv1BN.average_op,
            self._conv1BN.average_op,
            self._conv2BN.average_op,
            self._dense1BN.average_op,
            self._dense2BN.average_op,
        ]

    def forward(self, input_tensor, bn_pl, act):
        print("######################################################")
        print("##################### CNN ############################")
        print("######################################################")

        input_tensor = tf.expand_dims(input_tensor, axis=[0])
        input_tensor = tf.expand_dims(input_tensor, axis=[-1])

        print("input: ", input_tensor.get_shape().as_list())

        patches = crop_patches_from_image(input_tensor, self._n_channel, self._n_frequency)

        B, P, F, _, C = patches.get_shape().as_list()

        x = tf.reshape(patches, [B * P, F, F, C])

        print("x: ", x.get_shape().as_list())

        conv1 = self._conv1.forward(x)
        conv1 = act(conv1)
        print("conv1: ", conv1.get_shape().as_list(), self.H1, self.W1)
        conv1 = self._conv1BN.normalize(conv1, bn_pl)

        conv2 = self._conv2.forward(conv1)
        conv2 = act(conv2)
        print("conv2: ", conv2.get_shape().as_list(), self.H2, self.W2)
        conv2 = self._conv2BN.normalize(conv2, bn_pl)

        deconv1 = self._deconv1.forward(conv2)
        deconv1 = act(deconv1)
        deconv1 = self._dconv1BN.normalize(deconv1, bn_pl)

        print("conv transpose1: ", deconv1.get_shape().as_list())

        deconv2 = self._deconv2.forward(deconv1)
        print("conv transpose2: ", deconv2.get_shape().as_list())

        # gate, sigmoid
        deconv2 = sigmoid(deconv2)

        flatten = tf.reshape(conv2, [-1, self.H2 * self.W2 * self._n_filters_conv2])
        print("flatten: ", flatten.get_shape().as_list())

        fc1 = self._dense1.forward(flatten)
        fc1 = self._dense1BN.normalize(fc1, bn_pl)
        fc1 = act(fc1) + fc1
        print("fc1: ", fc1.get_shape().as_list())

        fc2 = self.dense2.forward(fc1)
        fc2 = self._dense2BN.normalize(fc2, bn_pl)
        fc2 = act(fc2) + fc2
        print("fc2: ", fc2.get_shape().as_list())

        attention = self._attention.forward(fc2)
        attention = tf.reshape(attention, [-1])
        print("attention: ", attention.get_shape().as_list())

        mask = generate_mask_via_attention(attention, self._n_channel, self._n_frequency)
        print("mask: ", mask.get_shape().as_list())
        gate = generate_mask_via_gate(deconv2, self._n_channel)
        gate = tf.squeeze(gate)
        print("gate: ", gate.get_shape().as_list())
        image = mask * gate

        return image

    @property
    def dict(self):
        return self.__dict__


class GraphConvolutionModule(object):
    def __init__(self,
                 name,
                 n_input,
                 nic,
                 npf,
                 npo,
                 n_dims_one,
                 n_dims_two):
        self.name = name

        self._n_input = n_input

        self.nic = nic
        self.npf = npf
        self.npo = npo

        self.n_dims_one = n_dims_one
        self.n_dims_two = n_dims_two

        self._n_input_dense = self._n_input

        with tf.variable_scope(self.name):
            self.gc_layer = GraphConvolutionLayer(
                "conv",
                self.nic,
                self.npf,
                self.npo
            )

            self.batch_norm_layer_gc = BatchNormLayer("convBN",
                                                      self.npf,
                                                      trainable=False
                                                      )
            self.pooling_layer_gc = PoolingOverFiltersLayer(
                "pooling",
                self.npf
            )

            self.affine_layer_one = DenseLayer("dense1",
                                               self._n_input_dense, self.n_dims_one,
                                               trainable_biases=False
                                               )
            self.batch_norm_layer_one = BatchNormLayer("dense1BN",
                                                       self.n_dims_one,
                                                       trainable=False
                                                       )
            self.affine_layer_two = DenseLayer("dense2",
                                               self.n_dims_one, self.n_dims_two,
                                               trainable_biases=False
                                               )
            self.batch_norm_layer_two = BatchNormLayer("dense2BN",
                                                       self.n_dims_two,
                                                       trainable=False
                                                       )
            self.affine_layer_logits = DenseLayer("dense_logits",
                                                  self.n_dims_two, 2,
                                                  trainable_biases=False
                                                  )

            self.bn_update_ops = [
                self.batch_norm_layer_one.average_op,
                self.batch_norm_layer_two.average_op,
                self.batch_norm_layer_gc.average_op
            ]

    def forward(self, x, L, bn_pl, act):
        flow = self.gc_layer.forward(x, L)
        flow = self.batch_norm_layer_gc.normalize(flow, bn_pl)
        flow = act(flow) + flow

        flow = self.pooling_layer_gc.forward(flow)
        flow = act(flow) + flow

        flow = self.affine_layer_one.forward(flow)
        flow = self.batch_norm_layer_one.normalize(flow, bn_pl)
        flow = act(flow) + flow

        flow = self.affine_layer_two.forward(flow)
        flow = self.batch_norm_layer_two.normalize(flow, bn_pl)
        flow = act(flow) + flow

        logits = self.affine_layer_logits.forward(flow)
        logits = tf.squeeze(logits)

        return logits, flow

    @property
    def dict(self):
        return self.__dict__
