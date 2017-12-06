#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-25, 11:55

@Description:

@Update Date: 17-11-25, 11:55
"""

from pro.config import Config
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, Adadelta, Nadam, RMSprop
from pro.dl import rmse, mape, mae, get_model_save_path
import os
from pro.dataset import get_train_test_data
from pro.dl.Factory import factory
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="the config path")
    args = parser.parse_args()

    print args.config

    # 配置文件
    conf = Config(args.config)

    # 数据读取
    data, arm_shape, train_xs, train_ys, train_arms, test_xs, test_ys, test_arms = \
        get_train_test_data(conf,
                            need_road_network_structure_matrix=conf.use_loopup,
                            no_adjacent_fill_zero=conf.no_adjacent_fill_zero)

    if conf.use_loopup:
        train_xs = [train_xs, train_arms]
        test_xs = [test_xs, test_arms]

    # model weights save path
    model_save_path = get_model_save_path(conf)
    # get model
    model = factory.get_model(conf, arm_shape)

    if conf.use_cache_model and os.path.exists(model_save_path):
        model.load_weights(model_save_path)
    else:
        adam = Adam(lr=conf.learning_rate)
        model.compile(adam, loss="mean_squared_error", metrics=["mae", "mape"])
        early_stopping = EarlyStopping(monitor="val_loss",
                                       patience=conf.early_stopping)

        check_points = ModelCheckpoint(model_save_path,
                                       monitor="val_loss",
                                       save_best_only=True,
                                       save_weights_only=True)
        model.summary()

        # log_filepath = "./keras_log"
        # tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
        # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的
        # 权值，每层输出值的分布直方图

        # 训练
        history = model.fit(train_xs,
                            train_ys,
                            epochs=conf.epochs,
                            batch_size=conf.batch_size,
                            callbacks=[early_stopping, check_points],
                            validation_data=[test_xs, test_ys])

        model.load_weights(model_save_path)

    # 测试
    predict = model.predict(test_xs, batch_size=conf.batch_size)

    predict = data.min_max_scala.inverse_transform(predict)
    y_true = data.min_max_scala.inverse_transform(test_ys)

    print "RMSE:", rmse(predict, y_true)
    print "MAE:", mae(predict, y_true)
    print "MAPE:", mape(predict, y_true)

    print "test"
    print " predict"
    print predict
    print
    print " real"
    print y_true
    print


if __name__ == '__main__':
    main()
