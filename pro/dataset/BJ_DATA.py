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
    def __init__(self, _min=-1, _max=1):
        self._min = _min
        self._max = _max
        assert (self._max > self._min)
        self.min = 0
        self.max = 0
        self.s = 0
        self.x = 0
        self.is_fit = False

    def fit(self, x):
        if self.is_fit is False:
            self.min = np.min(x)
            self.max = np.max(x)
            self.z = (self.max - self.min) / float((self._max - self._min))
            self.x = (self.min + self.max - self.z * (self._min + self._max)) / 2
            self.is_fit = True

    def transform(self, x):
        if not self.is_fit:
            print "please fit first"
            exit(1)
        _x = x.copy()
        _x = (_x - self.x).astype(float) / self.z
        return _x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if not self.is_fit:
            print "please fit first"
            exit(1)
        _x = x.copy()
        _x = _x.astype(float) * self.z + self.x
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
                 stride_sparse=False, stride_edges=1,
                 fix_adjacent_road_num=-1):
        stm, arm, t = completion_data(path, suffix,
                                      start_hour=start_hour,
                                      end_hour=end_hour,
                                      time_fill_split=time_fill_split,
                                      road_fill_split=road_fill_split,
                                      stride_sparse=stride_sparse,
                                      stride_edges=stride_edges,
                                      A=fix_adjacent_road_num)

        self.stm = stm
        self.arm = arm
        self.t = t

        stm = stm[:] * 3.6
        stm = self.min_max_scala.fit_transform(stm)
        xs = []
        ys = []
        _i = 0
        _start = 0
        current = ""
        length = self.observe_length + self.predict_length

        while _i < stm.shape[1]:
            if t[_i][:8] == current:
                _i += 1
            else:
                if _i != _start:
                    smooth_part = stm[:, _start: _i - 1].copy()
                    for _j in range(1, smooth_part.shape[1] - 1):
                        smooth_part[:, _j] = stm[:, _start: _i - 1][:, _j - 1] * 0.15 + stm[:, _start: _i - 1][:, _j] * 0.7 + stm[:, _start: _i - 1][:, _j + 1] * 0.15
                    smooth_part[:,0] = stm[:, _start: _i - 1][:, 1] * 0.2 + stm[:, _start: _i - 1][:, 0] * 0.8
                    smooth_part[:,smooth_part.shape[1] - 1] = stm[:, _start: _i - 1][:, smooth_part.shape[1] - 2] * 0.2 + stm[:, _start: _i - 1][:, smooth_part.shape[1] - 1] * 0.8
                    for _k in range(0, smooth_part.shape[1] - length + 2):
                        xs.append(stm[:, _k:_k + self.observe_length])
                        ys.append(stm[:, _k + self.observe_length:_k + self.observe_length + self.predict_length])
                _start = _i
                current = t[_i][:8]
                _i += 1

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
