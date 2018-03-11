import tensorflow as tf
from .base import BaseNet


class Net(BaseNet):

    def __init__(self, n_feats, weight_decay=0.0001, name='resnet-20'):
        super().__init__(name)

        self.data_format = 'channels_first'
        self.n_feats = n_feats
        self.weight_decay = weight_decay

    def block(self, inputs, filters, stride, name, conv_args={}, bn_args={}):
        with tf.variable_scope(name):
            outputs = tf.layers.conv2d(inputs, filters, 3, stride, name='conv1', **conv_args)
            outputs = tf.layers.batch_normalization(outputs, name='conv1/bn', **bn_args)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(outputs, filters, 3, 1, name='conv2', **conv_args)
            outputs = tf.layers.batch_normalization(outputs, name='conv2/bn', **bn_args)

            if stride == 1:
                shortcut = inputs
            else:
                shortcut = tf.layers.conv2d(
                    inputs, filters, 1, stride, name='shortcut', **conv_args)
                shortcut = tf.layers.batch_normalization(
                    shortcut, name='shortcut/bn', **bn_args)
            
            return tf.nn.relu(outputs + shortcut)

    def call(self, images, is_training):
        w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(2),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)}
        conv_args = {
            'data_format': self.data_format,
            'use_bias': False,
            'padding': 'same',
            **w_args}
        bn_args = {
            'training': is_training,
            'fused': True,
            'scale': True,
            'axis': 1}
        pool_args = {
            'data_format': self.data_format}
        fc_args = {
            'kernel_initializer': tf.initializers.truncated_normal(0, stddev=0.001),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)}

        struct = ([2, 2, 2], [16, 32, 64])

        net = tf.layers.conv2d(images, 16, 3, 1, name='conv1', **conv_args)
        net = tf.layers.batch_normalization(net, name='conv1/bn', **bn_args)
        net = tf.nn.relu(net)

        for i, (n_block, n_filters) in enumerate(zip(struct[0], struct[1])):
            stride = 1 if i == 0 else 2
            for j in range(n_block):
                stride = stride if j == 0 else 1
                net = self.block(
                    net, n_filters, stride,
                    name='res{:d}_{:d}'.format(i + 1, j + 1),
                    conv_args=conv_args, bn_args=bn_args)

        shape = net.shape.as_list()
        net = tf.layers.average_pooling2d(
            net, shape[2:], 1,name='global_avg_pool', **pool_args)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, self.n_feats, name='fc1', **fc_args)
        net = tf.nn.l2_normalize(net, axis=1)
        return net
