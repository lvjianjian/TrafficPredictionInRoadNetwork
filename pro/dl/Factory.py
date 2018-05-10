#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-30, 15:22

@Description:

@Update Date: 17-11-30, 15:22
"""

from keras.layers import Input, Dense, Activation, Embedding, Flatten, Reshape, Layer, Dropout, BatchNormalization
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.merge import Add, Concatenate
from keras.layers.local import LocallyConnected2D
from keras.models import Model
from pro.dl import rmse, mape, mae, MyReshape, MyInverseReshape, get_model_save_path, matrixLayer
from pro.dl.LookupConv import Lookup, LookUpSqueeze


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
        output = Dense(conf.predict_length, activation="tanh")(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def BiRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        # output = SimpleRNN(32, return_sequences=True)(output)
        output1 = SimpleRNN(conf.observe_length)(output)
        output2 = SimpleRNN(conf.observe_length, go_backwards=True)(output)
        output = Add()([output1, output2])
        # output = Dropout(0.1)(output)
        output = Dense(conf.predict_length, activation="tanh")(output)
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

    def DCNN_model(self, conf, arm_shape):
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
        output = Activation(activation="tanh")(output)

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
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        # f = Flatten()(pool3)
        # output = Dense(road_num * conf.predict_length, activation="sigmoid")(f)
        # output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    # def LCNN_model(self, conf, arm_shape):
    #     road_num = arm_shape[0]
    #     A = arm_shape[1]
    #     input_x = Input((road_num, conf.observe_length, 1))
    #     input_ram = Input(arm_shape)
    #
    #     output = Lookup(conf.batch_size)([input_x, input_ram])
    #     output = Conv3D(32, (1, 2, 2), activation="relu", padding="same")(output)
    #     output = Conv3D(32, (1, 2, 2), activation="relu", padding="same")(output)
    #     output = MaxPooling3D((1, A, 2))(output)
    #     output = LookUpSqueeze()(output)
    #
    #     output = Lookup(conf.batch_size)([output, input_ram])
    #     output = Conv3D(16, (1, 2, 2), activation="relu", padding="same")(output)
    #     output = Conv3D(16, (1, 2, 2), activation="relu", padding="same")(output)
    #     output = MaxPooling3D((1, A, 2))(output)
    #     output = LookUpSqueeze()(output)
    #
    #     output = Conv2D(1, (1, 2), activation="sigmoid")(output)
    #     output = Reshape((road_num, conf.predict_length))(output)
    #     model = Model(inputs=[input_x, input_ram], outputs=output)
    #     return model

    def LCRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape(conf.batch_size)(output)

        model = Model(inputs=inputs, outputs=output)
        return model

    def __E_input_output(self, conf, arm_shape, activation="tanh"):
        road_num = arm_shape[0]
        if conf.observe_p != 0:
            input_x1 = Input((road_num, conf.observe_p))
            output1 = MyReshape(conf.batch_size)(input_x1)
            output1 = Dense(conf.observe_p + 1, activation="relu")(output1)

        if conf.observe_t != 0:
            input_x2 = Input((road_num, conf.observe_t))
            output2 = MyReshape(conf.batch_size)(input_x2)
            output2 = Dense(conf.observe_t + 1, activation="relu")(output2)

        if conf.observe_p != 0:
            if conf.observe_t != 0:
                output = Concatenate()([output1, output2])
                input_x = [input_x1, input_x2]
            else:
                output = output1
                input_x = input_x1
        else:
            output = output2
            input_x = input_x2

        output = Dense(conf.predict_length, activation=activation)(output)
        output = MyInverseReshape(conf.batch_size)(output)

        input_x3 = Input((conf.predict_length, 37))  # 37 is externel dim
        if isinstance(input_x, list):
            input_x += [input_x3]
        else:
            input_x = [input_x, input_x3]

        output_3 = MyReshape(conf.batch_size)(input_x3)
        output_3 = Dense(road_num, activation=activation)(output_3)
        output_3 = MyInverseReshape(conf.batch_size)(output_3)
        output_3 = Reshape((road_num, conf.predict_length))(output_3)
        output = Add()([output, output_3])
        return input_x, output

    def E_model(self, conf, arm_shape):
        input_x, output = self.__E_input_output(conf, arm_shape)
        model = Model(inputs=input_x, output=output)
        return model

    def LCRNNBN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        inputs = [input_x, input_ram]
        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=inputs, outputs=output)
        return model

    def LCNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        # input_effective = Input((arm_shape[0],))
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])
        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Conv2D(1, (1, 5), activation="relu")(output)
            output = Reshape((road_num, conf.predict_length))(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Conv2D(1, (1, 5), activation="tanh")(output)
            output = Reshape((road_num, conf.predict_length))(output)

        model = Model(inputs=inputs, outputs=output)
        return model


factory = Factory()
