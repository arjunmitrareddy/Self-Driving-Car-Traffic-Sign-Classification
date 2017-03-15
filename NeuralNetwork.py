import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as C
class NeuralNetwork:

    def __init__(self, x):
        self.current_layer = None
        self.x = x

    def convolution_layer(self, output_depth, filter_width, stride, padding, scope, initial_layer=False):
        if initial_layer:
            self.current_layer = slim.conv2d(self.x, output_depth, [filter_width, filter_width], [stride,stride], scope=scope, padding=padding)
        else:
            self.current_layer = slim.conv2d(self.current_layer, output_depth, [filter_width, filter_width], stride, scope=scope, padding=padding)
        return self

    def max_pool_layer(self, filter_width, stride, padding, scope):
        self.current_layer = slim.max_pool2d(self.current_layer, [filter_width, filter_width], stride, padding=padding, scope=scope)
        return self

    def inception_layer(self, output_depth):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
            ic1_1 = slim.conv2d(self.current_layer, output_depth, [1, 1], stride=5, scope='ic1_1')

            ic2_1 = slim.conv2d(self.current_layer, output_depth, [1, 1], stride=5, scope='ic2_1')
            ic2_2 = slim.conv2d(ic2_1, output_depth, [3, 3], stride=1, scope='ic2_2')

            ic3_1 = slim.conv2d(self.current_layer, output_depth, [1, 1], stride=5, scope='ic3_1')
            ic3_2 = slim.conv2d(ic3_1, output_depth, [5, 5], stride=1, scope='ic3_2')

            ic4_1 = slim.max_pool2d(self.current_layer, [1, 1], 1, padding='SAME',scope='ic4_1')
            ic4_2 = slim.conv2d(self.current_layer, output_depth, [1, 1], scope='ic4_1')

            inception_layer = {
                'layer1': [ic1_1],
                'layer2': [ic2_1, ic2_2],
                'layer3': [ic3_1, ic3_2],
                #'layer4': [ic4_1_reshape, ic4_2_reshape]
            }
            inception_layers = []
            for group in inception_layer.values():
                for layer in group:
                    inception_layers.append(layer)
            self.current_layer = tf.concat(inception_layers, 1)
        return self

    def flatten(self):
        self.current_layer = tf.contrib.layers.flatten(self.current_layer)
        return self

    def fully_connected_layer(self, output_depth, scope):
        self.current_layer = slim.fully_connected(self.current_layer, output_depth, scope=scope)
        return self

    def dropout_layer(self):
        self.current_layer = tf.nn.dropout(self.current_layer, keep_prob=C.KEEP_PROB)
        return self


