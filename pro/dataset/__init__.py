#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-3, 09:35

@Description:

@Update Date: 17-11-3, 09:35
"""

from BJ_DATA import BJ_DATA
import numpy as np


def get_multiple_equal_batch_size(datas, batch_size):
    n = datas[0].shape[0]
    n = n / batch_size
    n = n * batch_size
    datas = [_d[-n:] for _d in datas]
    return datas


def get_train_test_data(conf, need_road_network_structure_matrix, no_adjacent_fill_zero):
    data = BJ_DATA(conf.observe_length, conf.predict_length, conf)
    xs, ys, arm = data.get_data(conf.data_path, conf.suffix + "_" + str(conf.time_window),
                                start_hour=conf.start_hour,
                                end_hour=conf.end_hour,
                                no_adjacent_fill_zero=no_adjacent_fill_zero,
                                time_fill_split=conf.time_fill_split, road_fill_split=conf.road_fill_split,
                                stride_sparse=conf.stride_sparse, stride_edges=conf.stride_edges,
                                fix_adjacent_road_num=conf.fix_adjacent_road_num)

    arm_shape = arm.shape

    xp = xs[1]
    xt = xs[2]
    xs = xs[0]


    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)

    arms = np.tile(arm, (xs.shape[0], 1, 1))
    train_xs, test_xs, train_ys, test_ys, train_arms, test_arms,\
        train_xp,test_xp,train_xt,test_xt = data.split(conf.test_ratio, [xs, ys, arms, xp,xt])

    train_xs, train_ys, train_arms,train_xp,train_xt = get_multiple_equal_batch_size([train_xs,
                                                                    train_ys,
                                                                    train_arms,
                                                                    train_xp,
                                                                    train_xt],
                                                                   conf.batch_size)

    test_xs, test_ys, test_arms, test_xp, test_xt = get_multiple_equal_batch_size([test_xs,
                                                                 test_ys,
                                                                 test_arms,
                                                                 test_xp,
                                                                 test_xt],
                                                                conf.batch_size)
    if need_road_network_structure_matrix:
        return data, arm_shape, train_xs, train_ys, train_arms,train_xp,train_xt,\
                                test_xs, test_ys, test_arms,test_xp,test_xt
    else:
        return data, arm_shape, train_xs, train_ys, None,train_xp,train_xt,\
               test_xs, test_ys, None, test_xp,test_xt
