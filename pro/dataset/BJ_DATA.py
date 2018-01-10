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
    def __init__(self, observe_length, predict_length, conf):
        self.data_name = "BJ"
        self.observe_length = observe_length
        self.predict_length = predict_length
        self.min_max_scala = MinMaxScalar()
        self.conf = conf
        self.observe_p = self.conf.observe_p
        self.observe_t = self.conf.observe_t

    @performance
    def get_data(self, path, suffix,
                 start_hour=8, end_hour=22,
                 time_fill_split=0.5, road_fill_split=0.2,
                 no_adjacent_fill_zero=True,
                 stride_sparse=False, stride_edges=1,
                 fix_adjacent_road_num=-1):

        basic_path = os.path.join(path, "cache", "{}_" + "{}_{}_{}_{}_{}_{}_{}".format(suffix,
                                                                                       start_hour,
                                                                                       end_hour,
                                                                                       time_fill_split,
                                                                                       road_fill_split,
                                                                                       self.conf.observe_length,
                                                                                       self.conf.observe_p))
        xc_path = basic_path.format("XC")
        xp_path = basic_path.format("XP")
        xt_path = basic_path.format("XT")
        ys_path = basic_path.format("YS")
        arm_path = basic_path.format("ARM")
        min_max_path = basic_path.format("MIN_MAX")
        E_path = basic_path.format("E")
        if not os.path.exists(xc_path + ".npy"):

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
            _i = 0
            _start = 0
            current = ""
            while _i < stm.shape[1]:
                if t[_i][:8] == current:
                    _i += 1
                else:
                    if _i != _start:
                        smooth_part = stm[:, _start: _i - 1].copy()
                        for _j in range(1, smooth_part.shape[1] - 1):
                            smooth_part[:, _j] = stm[:, _start: _i - 1][:, _j - 1] * 0.15 + stm[:, _start: _i - 1][:,
                                                                                            _j] * 0.7 + stm[:,
                                                                                                        _start: _i - 1][
                                                                                                        :,
                                                                                                        _j + 1] * 0.15
                        smooth_part[:, 0] = stm[:, _start: _i - 1][:, 1] * 0.2 + stm[:, _start: _i - 1][:, 0] * 0.8
                        smooth_part[:, smooth_part.shape[1] - 1] = stm[:, _start: _i - 1][:,
                                                                   smooth_part.shape[1] - 2] * 0.2 + stm[:,
                                                                                                     _start: _i - 1][
                                                                                                     :,
                                                                                                     smooth_part.shape[
                                                                                                         1] - 1] * 0.8
                        stm[:, _start: _i - 1] = smooth_part
                        # for _k in range(0, smooth_part.shape[1] - length + 2):
                        #     xs.append(stm[:, _k:_k + self.observe_length])
                        #     ys.append(stm[:, _k + self.observe_length:_k + self.observe_length + self.predict_length])
                    _start = _i
                    current = t[_i][:8]
                    _i += 1

            # externel data
            holiday = load_holiday(t, os.path.join(path, "BJ_Holiday.txt"))
            meteorol = load_meteorol(t, os.path.join(path, "BJ_WEATHER.h5"))
            vec = timestamp2vec(t)

            externel_data = np.hstack([holiday,meteorol,vec])

            tt = []
            for _t in self.t:
                tt.append(pd.to_datetime(_t))

            time_dict = dict(zip(tt, range(len(tt))))
            T = 24 * 60 / self.conf.time_window
            offset_frame = pd.DateOffset(minutes=self.conf.time_window)

            XC = []
            XP = []
            XT = []
            YS = []
            E = []
            for _t in tt:
                indexs = []
                not_it = False
                for _i in range(self.observe_length, 0, -1):
                    _tt = _t - _i * offset_frame
                    if (_tt in time_dict):
                        indexs.append(time_dict[_tt])
                    else:
                        not_it = True
                if not_it:
                    continue
                xc = stm[:, indexs]
                # print indexs
                indexs = []
                for _i in range(self.observe_p, 0, -1):
                    _tt = _t - _i * T * offset_frame
                    if (_tt in time_dict):
                        indexs.append(time_dict[_tt])
                    else:
                        not_it = True
                if not_it:
                    continue
                xp = stm[:, indexs]
                # print indexs
                indexs = []
                for _i in range(self.observe_t, 0, -1):
                    _tt = _t - _i * T * 7 * offset_frame
                    if (_tt in time_dict):
                        indexs.append(time_dict[_tt])
                    else:
                        not_it = True
                if not_it:
                    continue
                # print indexs
                xt = stm[:, indexs]

                indexs = []
                for _i in range(self.predict_length):
                    _tt = _t + _i * offset_frame
                    if (_tt in time_dict):
                        indexs.append(time_dict[_tt])
                    else:
                        not_it = True
                if not_it:
                    continue
                y = stm[:, indexs]
                # print indexs

                E.append(externel_data[indexs])
                XC.append(xc)
                XP.append(xp)
                XT.append(xt)
                YS.append(y)

            XC = np.stack(XC, axis=0)
            XP = np.stack(XP, axis=0)
            XT = np.stack(XT, axis=0)
            YS = np.stack(YS, axis=0)
            E = np.stack(E,axis=0)
            # print XC.shape
            # print XP.shape
            # print XT.shape
            # print YS.shape
            if not no_adjacent_fill_zero:
                for _i in range(arm.shape[0]):
                    _a = arm[_i]
                    _a[_a[:] == arm.shape[0]] = _i

            np.save(xc_path, XC)
            np.save(xp_path, XP)
            np.save(xt_path, XT)
            np.save(ys_path, YS)
            np.save(arm_path, arm)
            np.save(E_path, E)
            cPickle.dump(self.min_max_scala, open(min_max_path, "w"))
        else:
            XC = np.load(xc_path + ".npy")
            XP = np.load(xp_path + ".npy")
            XT = np.load(xt_path + ".npy")
            YS = np.load(ys_path + ".npy")
            arm = np.load(arm_path + ".npy")
            E = np.load(E_path+".npy")
            self.min_max_scala = cPickle.load(open(min_max_path))
        # print XC.shape
        # print XP.shape
        # print XT.shape
        # print YS.shape
        # print arm.shape
        # print E.shape
        return [XC, XP, XT, E], YS, arm

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
