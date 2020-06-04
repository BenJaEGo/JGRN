import tensorflow as tf
import numpy as np


def get_variable_he(name, shape, dtype=tf.float32, trainable=True):
    var_weights = tf.get_variable(name=name,
                                  shape=shape,
                                  initializer=tf.initializers.he_normal(),
                                  dtype=dtype,
                                  trainable=trainable
                                  )
    return var_weights


def get_variable_constant(name, shape, offset=0., dtype=tf.float32, trainable=True):
    var_biases = tf.get_variable(name=name,
                                 shape=shape,
                                 initializer=tf.constant_initializer(
                                     value=offset
                                 ),
                                 dtype=dtype,
                                 trainable=trainable
                                 )

    return var_biases


def sigmoid(x):
    return tf.nn.sigmoid(x)


def softsign(x):
    return tf.nn.softsign(x)


def relu(x):
    return tf.nn.relu(x)


def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def conv_transpose(input_tensor, filters, kernel_size, strides, padding, name=None):
    output_tensor = tf.layers.conv2d_transpose(inputs=input_tensor,
                                               filters=filters,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               name=name)
    return output_tensor


def elu(x):
    return tf.nn.elu(x)


def tanh(x):
    return tf.nn.tanh(x)


def softmax(x):
    return tf.nn.softmax(x)


def softplus(x):
    return tf.nn.softplus(x)


def max_pooling(input_tensor, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name=None):
    output_tensor = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding, name=name)
    return output_tensor


def laplacian(A, normalized=True):
    # Degree matrix
    d = tf.reduce_sum(A, axis=0)

    M = A.get_shape().as_list()[0]

    # Laplacian matrix.
    if normalized:
        d = tf.convert_to_tensor(1.0) / tf.sqrt(d + tf.convert_to_tensor(1e-6))
        D = tf.linalg.tensor_diag(d)
        I = tf.eye(M)
        L = I - tf.matmul(tf.matmul(D, A), D)
    else:
        D = tf.linalg.tensor_diag(d)
        L = D - A
    return L


def rescale_L(L, lmax=2):
    M = L.get_shape().as_list()[0]
    I = tf.eye(M)
    L /= lmax / 2
    L -= I
    return L


def chebyshev5(x, W, L, Fout, K):
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)

    L = rescale_L(L)

    x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
        return tf.concat([x, x_], axis=0)  # K x M x Fin*N

    if K > 1:
        x1 = tf.matmul(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.matmul(L, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
    x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filter bank per feature pair.
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [N, M, Fout])  # N x M x Fout


def generate_mask_via_attention(attention, n_channel=6, n_frequency=114):
    n_patches = attention.shape[0]

    P = []
    [P.append(tf.ones([n_frequency, n_frequency]) * attention[idx]) for idx in range(n_patches)]
    P = tf.convert_to_tensor(P)

    rows = []
    for i in range(n_channel):
        row = []
        for j in range(n_channel):
            row.append(P[i * n_channel + j])

        row = tf.concat(row, axis=1)
        rows.append(row)
    mask = tf.concat(rows, axis=0)

    return mask


def generate_mask_via_gate(P, n_channel):
    rows = []
    for i in range(n_channel):
        row = []
        for j in range(n_channel):
            row.append(P[i * n_channel + j])

        row = tf.concat(row, axis=1)
        rows.append(row)
    mask = tf.concat(rows, axis=0)

    return mask


def crop_patches_from_image(images, n_channel, n_frequency):
    B, H, W, C = images.get_shape().as_list()
    row1 = np.repeat(np.arange(n_channel).reshape([-1, 1]), repeats=n_channel, axis=0).squeeze()
    row2 = np.repeat(np.arange(n_channel).reshape([-1, 1]), repeats=n_channel, axis=1).transpose().flatten()
    row3 = np.repeat(np.arange(n_channel).reshape([-1, 1]), repeats=n_channel, axis=0).squeeze() + 1
    row4 = np.repeat(np.arange(n_channel).reshape([-1, 1]), repeats=n_channel, axis=1).transpose().flatten() + 1

    locations = np.stack((row1, row2, row3, row4), axis=1)

    locations = locations / n_channel
    locations = np.array(locations, np.float32)
    locations = tf.convert_to_tensor(locations)

    patches = []
    crop_H = n_frequency
    crop_W = n_frequency
    for idx in range(n_channel ** 2):
        loc = locations[idx]
        boxes = tf.reshape(tf.tile(loc, [B]), [B, 4])
        box_indices = tf.range(0, B)
        crop_size = tf.convert_to_tensor([crop_H, crop_W])
        crop_size = tf.cast(crop_size, tf.int32)

        crops = tf.image.crop_and_resize(images, boxes, box_indices, crop_size)

        patches.append(crops)

    patches = tf.convert_to_tensor(patches, tf.float32)
    patches = tf.transpose(patches, [1, 0, 2, 3, 4])

    return patches


def min_max_scale(tensor):
    tensor = tf.div(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )
    return tensor
