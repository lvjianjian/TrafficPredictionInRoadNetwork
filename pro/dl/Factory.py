#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-30, 15:22

@Description:

@Update Date: 17-11-30, 15:22
"""

from keras.layers import Input, Dense, Activation, Embedding, Flatten, Reshape, Layer, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Model
from pro.dl import rmse, mape, mae, MyReshape, MyInverseReshape, get_model_save_path, Lookup, LookUpSqueeze


# def resnet(input):
#     output
#
#     return output

class Factory(object):
    def get_model(self, conf, arm_shape):
        print "use model ", conf.model_name
        model = None
        function_name = "{}_model(conf, arm_shape)".format(conf.model_name)
        exec "model = self." + function_name
        return model

    def RNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        # output = SimpleRNN(32, return_sequences=True)(output)
        output = SimpleRNN(conf.observe_length)(output)
        # output = Dropout(0.1)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def GRU_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        output = GRU(conf.observe_length)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def LSTM_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        output = LSTM(conf.observe_length)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def CNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))

        output = Conv2D(32, (5, 2), strides=(1, 1), padding="same")(input_x)
        output = Conv2D(32, (5, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="sigmoid")(output)

        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="sigmoid")(output)

        output = Conv2D(1, (2, 2), strides=(1, 1), padding="same")(output)
        output = Conv2D(1, (2, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="sigmoid")(output)

        output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def CRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = Conv2D(32, (2, 2), strides=(1, 1), padding="same")(input_x)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="relu")(output)
        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        # pool2 = AveragePooling2D(pool_size=(1,2))(conv2)
        # pool2 = Activation(activation="sigmoid")(conv2)
        # conv3 = Conv2D(4, (2, 2), strides=(1, 1), padding="same")(pool2)
        # pool3 = AveragePooling2D(pool_size=(1, 2))(conv3)
        output = Activation(activation="relu")(output)
        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        output = Dense(1)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        # f = Flatten()(pool3)
        # output = Dense(road_num * conf.predict_length, activation="sigmoid")(f)
        # output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def LCNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)

        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(32, (1, 2, 2), activation="relu", padding="same")(output)
        output = Conv3D(32, (1, 2, 2), activation="relu", padding="same")(output)
        output = MaxPooling3D((1, A, 2))(output)
        output = LookUpSqueeze()(output)


        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, 2, 2), activation="relu", padding="same")(output)
        output = Conv3D(16, (1, 2, 2), activation="relu", padding="same")(output)
        output = MaxPooling3D((1, A, 2))(output)
        output = LookUpSqueeze()(output)

        output = Conv2D(1, (1, 2), activation="sigmoid")(output)
        output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=[input_x, input_ram], outputs=output)
        return model



    def LCRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)

        output = Lookup(conf.batch_size)([input_x, input_ram])
        # glorot_normal he_uniform
        output = Conv3D(32, (1, A, 2), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(32, (1, A, 2), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        output = LookUpSqueeze()(output)
        #
        # output = Lookup(batch_size)([output, input_ram])
        # output = Conv3D(32, (1, A, 5), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        # output = LookUpSqueeze()(output)


        output = MyReshape(conf.batch_size)(output)
        output = GRU(32, activation="sigmoid")(output)
        output = Dense(1)(output)
        output = MyInverseReshape(conf.batch_size)(output)

        # output = Lookup(batch_size)([output, input_ram])
        # output = Conv3D(32, (1, A, 3), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        # output = LookUpSqueeze()(output)
        #
        # output = Lookup(batch_size)([output, input_ram])
        # output = Conv3D(16, (1, A, 3), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        # output = LookUpSqueeze()(output)
        #
        #
        # output = Lookup(batch_size)([output, input_ram])
        # output = Conv3D(1, (1, A, 4), activation="sigmoid", kernel_initializer="glorot_uniform")(output)
        # output = LookUpSqueeze()(output)

        # output = Lookup(batch_size)([output, input_ram])
        # output = Conv3D(1, (1, A, 3), activation="tanh")(output)
        # output = Squeeze()(output)

        # output = Flatten()(output)
        # output = Dense(road_num * conf.predict_length, activation="sigmoid")(output)
        # output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=[input_x, input_ram], outputs=output)
        return model

    def LCNN_model_test(conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        input_effective = Input((arm_shape[0],))
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(32, (1, A, 2), activation="relu")(output)
        # output = Conv3D(32, (1, 2, 2), activation="relu", padding="same")(output)
        # output = MaxPooling3D((1, A, 2))(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])


        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        # # output = Conv3D(16, (1, 2, 2), activation="relu", padding="same")(output)
        # # output = MaxPooling3D((1, A, 2))(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])

        output = Conv2D(1, (1, 8), activation="sigmoid")(output)
        output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=[input_x, input_ram, input_effective], outputs=output)
        return model


factory = Factory()
