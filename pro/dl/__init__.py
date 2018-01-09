#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-23, 15:00

@Description:

@Update Date: 17-11-23, 15:00
"""

import numpy as np
from keras.layers import Layer
import tensorflow as tf
import os
import keras.backend as K


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def mape(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))
    return np.mean(diff)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def get_model_save_path(conf):
    model_path = os.path.join(conf.model_path, str(conf.time_window))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_save_path = os.path.join(model_path, "{}_{}_{}_{}_{}_{}_{}.best.h5".format(conf.config_name,
                                                                                     conf.suffix,
                                                                                     conf.time_window,
                                                                                     conf.time_fill_split,
                                                                                     conf.road_fill_split,
                                                                                     conf.stride_edges,
                                                                                     conf.model_name))
    return model_save_path


class MyReshape(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(MyReshape, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        return tf.reshape(inputs, (self.batch_size * input_shape[1],) + input_shape[2:])

    def compute_output_shape(self, input_shape):
        return (self.batch_size * input_shape[1],) + input_shape[2:]


class MyInverseReshape(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(MyInverseReshape, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        return tf.reshape(inputs, (self.batch_size, input_shape[0] / self.batch_size, input_shape[1]))

    def compute_output_shape(self, input_shape):
        return (self.batch_size, input_shape[0] / self.batch_size, input_shape[1])





# class MyRNN(Layer):
#     def __init__(self, batch_size, rnn_units, predict_length, **kwargs):
#         print "my rnn init"
#         self.batch_size = batch_size
#         self.rnn_units = rnn_units
#         self.predict_length = predict_length
#         super(MyRNN, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         print "my rnn build"
#         self.rnn = SimpleRNN(self.rnn_units)
#         self.dense = Dense(self.predict_length)
#
#     def call(self, inputs, **kwargs):
#         print "my rnn call"
#         outputs = []
#         for _i in range(self.batch_size):
#             output = self.rnn(inputs[_i])
#             output = self.dense(output)
#             outputs.append(output)
#         output = tf.stack(outputs, axis=0)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1], self.predict_length)

if __name__ == '__main__':
    from keras.models import Model
    from keras.layers import Input

    input_x = np.arange(10 * 5 * 3 * 1).reshape((10, 5, 3, 1))
    output_x = np.arange(10 * 5 * 3 * 1).reshape((10, 5, 3))
    input = Input((5, 3, 1))
    output = MyReshape(2)(input)
    output = MyInverseReshape(2)(output)
    model = Model(input, output)
    model.compile("sgd", "mse")

    model.summary()
    predict = model.predict(input_x, batch_size=2)
    print predict
    print
    # print output_x
    # print (predict - output_x).sum()
