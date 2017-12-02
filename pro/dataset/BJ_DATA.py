#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-23, 13:51

@Description:

@Update Date: 17-11-23, 13:51
"""

from pro.util import *


class MinMaxScalar(object):
    def __init__(self):
        self.min = 0
        self.max = 0
        self.is_fit = False

    def fit(self, x):
        self.min = np.min(x)
        self.max = np.max(x)
        self.is_fit = True

    def transform(self, x):
        if not self.is_fit:
            print "please fit first"
            exit(1)
        _x = x.copy()
        _x = (_x - self.min).astype(float) / (self.max - self.min)
        return _x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if not self.is_fit:
            print "please fit first"
            exit(1)
        _x = x.copy()
        _x = _x.astype(float) * (self.max - self.min) + self.min
        return _x


class BJ_DATA(object):
    def __init__(self, observe_length, predict_length):
        self.data_name = "BJ"
        self.observe_length = observe_length
        self.predict_length = predict_length
        self.min_max_scala = MinMaxScalar()

    @performance
    def get_data(self, path, suffix,
                 start_hour=8, end_hour=22,
                 time_fill_split=0.5, road_fill_split=0.2,
                 no_adjacent_fill_zero=True,
                 stride_sparse=False, stride_edges=1):
        stm, arm, t = completion_data(path, suffix,
                                      start_hour=start_hour,
                                      end_hour=end_hour,
                                      time_fill_split=time_fill_split,
                                      road_fill_split=road_fill_split,
                                      stride_sparse=stride_sparse,
                                      stride_edges=stride_edges)
        stm = stm[:] * 3.6
        stm = self.min_max_scala.fit_transform(stm)
        xs = []
        ys = []

        length = self.observe_length + self.predict_length
        for _i in range(stm.shape[1] - length + 1):
            xs.append(stm[:, _i:_i + self.observe_length])
            ys.append(stm[:, _i + self.observe_length:_i + self.observe_length + self.predict_length])
        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)

        if not no_adjacent_fill_zero:
            for _i in range(arm.shape[0]):
                _a = arm[_i]
                _a[_a[:] == arm.shape[0]] = _i

        return xs, ys, arm

    def split(self, test_ratio, datas):
        n = datas[0].shape[0]
        return_datas = []
        test_size = int(n * test_ratio)
        for _d in datas:
            return_datas.append(_d[:-test_size])
            return_datas.append(_d[-test_size:])

        return return_datas


if __name__ == '__main__':
    data = BJ_DATA(10, 2)
    xs, ys, arm = data.get_data("data/", "0123class_sub_region1_5")
