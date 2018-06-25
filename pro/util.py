#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-3, 09:35

@Description:

@Update Date: 17-11-3, 09:35
"""

import pandas as pd
import numpy as np
import os
import re
import scipy.sparse as sp
import cPickle
import time
import threading
import Queue
import h5py

region1 = [(116.274194, 39.832815), (116.501429, 39.998614)]  # 左下角 和 右上角 4,5环之间

sub_region1 = [(116.279512, 39.836306), (116.36, 39.908958)]  # 子图
sub_region2 = [(116.378254, 39.948574), (116.443794, 39.998559)]

CACHE = "cache"
REGEX = re.compile("\((.*),(.*),(.*)\)")


def performance(f):
    def fn(*args, **kwargs):
        start = time.time()
        r = f(*args, **kwargs)
        end = time.time()
        print "function {} cost {} s".format(f.__name__, (end - start))
        print
        return r

    return fn


def load_all_RG(path="data/R_G.csv"):
    rg = pd.read_csv(path,
                     sep=":",
                     names=["edge_id",
                            "s_id",
                            "e_id",
                            "direction",
                            "class",
                            "attribute",
                            "width",
                            "length",
                            "speed_limit",
                            "line_string"],
                     index_col=False)
    return rg


def load_all_RG_node(path="data/R_G_node.csv"):
    rg_node = pd.read_csv(path, index_col=0)
    return rg_node


def load_part_RG(path, suffix="part"):
    rg_path = os.path.join(path, "R_G_{}.csv".format(suffix))
    return pd.read_csv(rg_path, index_col=0)


def load_part_RG_node(path, suffix="part"):
    rg_node_path = os.path.join(path, "R_G_node_{}.csv".format(suffix))
    return pd.read_csv(rg_node_path, index_col=0)


def get_node_id2id_dict(rg_node):
    node_ids = rg_node.node_id.unique()
    node_id2id_dict = dict(zip(node_ids, range(len(node_ids))))
    return node_id2id_dict


def get_id2node_id_dict(node_id2id_dict):
    return dict(zip(node_id2id_dict.values(), node_id2id_dict.keys()))


def get_s_e_id_of_edge(edge_id, rg):
    se = rg[rg.edge_id == edge_id][["s_id", "e_id"]].values
    if len(se) == 0:
        return None, None
    se = se[0]
    return long(se[0]), long(se[1])


def split_one_link_avg_speed(l, nodeid2id_dict, r, rg):
    items = l.strip().split("|")
    one_time = items[0]
    rows = []
    datas = []
    cs = []
    for _i in range(1, len(items)):
        _v = items[_i]
        g = r.match(_v.strip())
        edgeid = long(g.group(1))
        avg_speed = float(g.group(2))
        traj_num = float(g.group(3))
        s_id, e_id = get_s_e_id_of_edge(edgeid, rg)
        if s_id is None or e_id is None:
            continue
        rows.append(nodeid2id_dict.get(s_id))
        cs.append(nodeid2id_dict.get(e_id))
        datas.append(avg_speed)
    return one_time, datas, rows, cs


def in_time(c, start_hour, end_hour):
    t = pd.to_datetime(c)
    if t.hour >= start_hour and t.hour < end_hour:
        return True
    else:
        return False


def get_time_day(c):
    return c[:8]


def load_raw_link_speed(path, suffix, cache=True, n_jobs=1):
    """

    :param path:
    :param suffix:
    :param cache:
    :param n_jobs:
    :return:
    """
    link_path = os.path.join(path, "link_avg_speed_" + suffix)
    rg = load_part_RG(path, suffix[:suffix.rindex("_")])
    rg_node = load_part_RG_node(path, suffix[:suffix.rindex("_")])
    nodeid2id_dict = get_node_id2id_dict(rg_node)
    time_list = []
    coo_matrix_list = []
    datass = []
    rowss = []
    columnss = []
    time_cache_path = os.path.join(path, CACHE, "time_{}.pkl".format(suffix))
    data_cache_path = os.path.join(path, CACHE, "data_{}.pkl".format(suffix))
    row_cache_path = os.path.join(path, CACHE, "row_{}.pkl".format(suffix))
    column_cache_path = os.path.join(path, CACHE, "column_{}.pkl".format(suffix))

    r = REGEX
    s = time.time()
    if cache and os.path.exists(time_cache_path):
        print time_cache_path, "exist"
        print "loading cache..."
        time_list = cPickle.load(open(time_cache_path, "r"))
        datass = cPickle.load(open(data_cache_path, "r"))
        rowss = cPickle.load(open(row_cache_path, "r"))
        columnss = cPickle.load(open(column_cache_path, "r"))
        print "load cache finish, spend {} s".format(time.time() - s)
    else:
        print "cache doesn't exist, loading from raw..."
        if n_jobs == 1:  # 单线程
            s = time.time()
            with open(link_path, "r") as f:
                for _index, l in enumerate(f):
                    if _index % 500 == 0:
                        print _index
                    one_time, datas, rows, cs = split_one_link_avg_speed(l, nodeid2id_dict, r, rg)
                    time_list.append(one_time)
                    datass.append(datas)
                    rowss.append(rows)
                    columnss.append(cs)
        else:  # 多线程
            queue = Queue.Queue(500)
            result = Queue.Queue(n_jobs)

            class worker(threading.Thread):
                def __init__(self, queue, result):
                    threading.Thread.__init__(self)
                    self.queue = queue
                    self.thread_stop = False
                    self.result = result
                    self.datass = []
                    self.rowss = []
                    self.times = []
                    self.columnss = []

                def run(self):
                    while not self.thread_stop:
                        try:
                            l = self.queue.get(block=True, timeout=20)
                        except Queue.Empty:
                            self.thread_stop = True
                            self.result.put([self.times, self.datass, self.rowss, self.columnss])
                            break
                        one_time, datas, rows, cs = split_one_link_avg_speed(l, nodeid2id_dict, r, rg)
                        self.datass.append(datas)
                        self.times.append(one_time)
                        self.rowss.append(rows)
                        self.columnss.append(cs)
                        self.queue.task_done()

            ws = []
            for _ in range(n_jobs):
                w = worker(queue, result)
                w.start()
                ws.append(w)
            with open(link_path, "r") as f:
                for l in f:
                    queue.put(l)
            for w in ws:  # waii all thread finish
                w.join()
            while not result.empty():
                try:
                    item = result.get(block=False)
                    time_list += item[0]
                    datass += item[1]
                    rowss += item[2]
                    columnss += item[3]
                except Queue.Empty:
                    break
            result.join()

        print "load from raw finish, spend {} s".format(time.time() - s)
        if cache:
            print "cache..."
            s = time.time()
            cPickle.dump(time_list, open(time_cache_path, "w"))
            cPickle.dump(datass, open(data_cache_path, "w"))
            cPickle.dump(rowss, open(row_cache_path, "w"))
            cPickle.dump(columnss, open(column_cache_path, "w"))
            print "cache finish, spend {} s".format(time.time() - s)

    for _t, _d, _r, _c in zip(time_list, datass, rowss, columnss):
        coo_matrix = sp.coo_matrix((_d, (_r, _c)), shape=(len(nodeid2id_dict), len(nodeid2id_dict)))
        coo_matrix_list.append(coo_matrix)
    return time_list, coo_matrix_list


def load_raw_link_speed_in_time(path, suffix, cache=True,
                                start_hour=8, end_hour=22,
                                remove_complete_day=True,
                                complete_ratio=0.9):
    time_interval = int(suffix.split("_")[-1])
    size = (end_hour - start_hour) * 60 / time_interval
    rg_node = load_part_RG_node(path, suffix[:suffix.rindex("_")])

    nodeid2id_dict = get_node_id2id_dict(rg_node)
    assert isinstance(start_hour, int)
    assert isinstance(end_hour, int)
    print "load_raw_link_speed_in_time ing.."
    time_cache_path = os.path.join(path, CACHE, "time_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    data_cache_path = os.path.join(path, CACHE, "data_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    row_cache_path = os.path.join(path, CACHE, "row_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    column_cache_path = os.path.join(path, CACHE, "column_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))

    nt = []
    ncoom = []
    if cache and os.path.exists(time_cache_path):
        s = time.time()
        print "cache exist"
        print "load cache..."
        nt = cPickle.load(open(time_cache_path, "r"))
        datass = cPickle.load(open(data_cache_path, "r"))
        rowss = cPickle.load(open(row_cache_path, "r"))
        columnss = cPickle.load(open(column_cache_path, "r"))
        for _t, _d, _r, _c in zip(nt, datass, rowss, columnss):
            coo_matrix = sp.coo_matrix((_d, (_r, _c)), shape=(len(nodeid2id_dict), len(nodeid2id_dict)))
            ncoom.append(coo_matrix)
        print "load cache finish, spend {} s".format(time.time() - s)
    else:
        print "cache not exist"
        s = time.time()
        t, coo_matrix = load_raw_link_speed(path, suffix, cache)
        for _t, _c in zip(t, coo_matrix):
            if in_time(_t, start_hour, end_hour):
                nt.append(_t)
                ncoom.append(_c)
        l = sorted(zip(nt, ncoom), key=lambda x: x[0])

        nt = []
        ncoom = []
        current_day = ""
        _nt = []
        _ncoom = []
        l += [("", "")]  # 可以让最后一天在循环中判断
        for _t, _m in l:
            _cd = get_time_day(_t)
            if _cd != current_day:
                # 判断之前数据的是否较为完整
                if remove_complete_day:
                    if len(_nt) > size * complete_ratio:
                        nt += _nt
                        ncoom += _ncoom
                        # print "stay {}, {}/{}".format(current_day,len(_nt),size)
                    else:
                        print "remove {}, just {}/{}".format(current_day, len(_nt), size)
                else:  # 不用去除
                    nt += _nt
                    ncoom += _ncoom
                _nt = [_t]
                _ncoom = [_m]
                current_day = _cd
            else:
                _ncoom.append(_m)
                _nt.append(_t)

        print "preprocess finish, spend {} s".format(time.time() - s)

        if cache:
            s = time.time()
            datass = []
            rowss = []
            columnss = []
            for _m in ncoom:
                datass.append(_m.data)
                rowss.append(_m.row)
                columnss.append(_m.col)
            cPickle.dump(nt, open(time_cache_path, "w"))
            cPickle.dump(datass, open(data_cache_path, "w"))
            cPickle.dump(rowss, open(row_cache_path, "w"))
            cPickle.dump(columnss, open(column_cache_path, "w"))
            print "cache finish, spend {} s".format(time.time() - s)
    return nt, ncoom


def get_edgeid2id(rg):
    edgeids = sorted(list(rg["edge_id"].unique()))
    # 将edgeid 映射到从0开始的连续key
    keys = range(len(edgeids))
    edgeid2id = dict(zip(edgeids, keys))
    return edgeid2id


def get_id2edgeid(rg):
    edgeids = sorted(list(rg["edge_id"].unique()))
    # 将edgeid 映射到从0开始的连续key
    keys = range(len(edgeids))
    id2edgeid = dict(zip(keys, edgeids))
    return id2edgeid


def split_one_link_avg_speed_by_road(l, regex, edgeid2id_dict, all_road_num):
    items = l.strip().split("|")
    one_time = items[0]
    datas = np.zeros((all_road_num,))
    traj_nums = np.zeros((all_road_num,))
    for _i in range(1, len(items)):
        _v = items[_i]
        g = regex.match(_v.strip())
        edgeid = long(g.group(1))
        avg_speed = float(g.group(2))
        traj_num = float(g.group(3))
        if edgeid in edgeid2id_dict:
            key = edgeid2id_dict[edgeid]
            datas[key] = avg_speed
            traj_nums[key] = traj_num
    return one_time, datas, traj_nums


@performance
def load_raw_link_speed_by_road(path, suffix, cache=True):
    link_path = os.path.join(path, "link_avg_speed_" + suffix)
    # rg = load_part_RG(path, suffix[:suffix.rindex("_")])
    rg = load_part_RG(path, suffix[:suffix.rindex("_")])
    edgeids = sorted(list(rg["edge_id"].unique()))
    all_road_num = len(edgeids)
    # 将edgeid 映射到从0开始的连续key
    edgeid2id = get_edgeid2id(rg)
    id2edgeid = get_id2edgeid(rg)

    # edgeid,speed,num 正则

    time_cache_path = os.path.join(path, CACHE, "time_by_road_{}.pkl".format(suffix))
    data_cache_path = os.path.join(path, CACHE, "data_by_road_{}.pkl".format(suffix))
    traj_num_cache_path = os.path.join(path, CACHE, "traj_num_by_road_{}.pkl".format(suffix))
    time_list = []
    data_list = []
    traj_num_list = []
    if cache and os.path.exists(time_cache_path):
        print time_cache_path, "exist"
        print "loading cache..."
        time_list = cPickle.load(open(time_cache_path, "r"))
        data_list = cPickle.load(open(data_cache_path, "r"))
        traj_num_list = cPickle.load(open(traj_num_cache_path, "r"))
        print "load cache finish"

    else:
        print "cache doesn't exist, loading from raw..."
        with open(link_path, "r") as f:
            for _index, l in enumerate(f):
                if _index % 500 == 0:
                    print _index
                one_time, datas, traj_nums = split_one_link_avg_speed_by_road(l, REGEX, edgeid2id, all_road_num)
                time_list.append(one_time)
                data_list.append(datas)
                traj_num_list.append(traj_nums)
        print "load from raw finish"

        if cache:
            print "cache..."
            cPickle.dump(time_list, open(time_cache_path, "w"))
            cPickle.dump(data_list, open(data_cache_path, "w"))
            cPickle.dump(traj_num_list, open(traj_num_cache_path, "w"))
            print "cache finish"
    return time_list, data_list, traj_num_list


@performance
def load_raw_link_speed_by_road_in_time(path, suffix, cache=True,
                                        start_hour=8, end_hour=22,
                                        remove_complete_day=True,
                                        complete_ratio=0.9):
    time_interval = int(suffix.split("_")[-1])
    size = (end_hour - start_hour) * 60 / time_interval  # 划分为多少个time interval

    assert isinstance(start_hour, int)
    assert isinstance(end_hour, int)
    print "load_raw_link_speed_by_road_in_time ing.."
    time_cache_path = os.path.join(path, CACHE, "time_by_road_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    data_cache_path = os.path.join(path, CACHE, "data_by_road_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    trajnum_cache_path = os.path.join(path, CACHE, "traj_num_by_road_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    nt = []
    nd = []
    nn = []
    if cache and os.path.exists(time_cache_path):
        print "cache exist"
        print "load cache..."
        nt = cPickle.load(open(time_cache_path, "r"))
        nd = cPickle.load(open(data_cache_path, "r"))
        nn = cPickle.load(open(trajnum_cache_path, "r"))
        print "load cache finish"
    else:
        print "cache not exist"
        t, datas, traj_nums = load_raw_link_speed_by_road(path, suffix, cache)
        for _t, _d, _n in zip(t, datas, traj_nums):
            if in_time(_t, start_hour, end_hour):
                nt.append(_t)
                nd.append(_d)
                nn.append(_n)
        l = sorted(zip(nt, nd, nn), key=lambda x: x[0])

        nt = []
        nd = []
        nn = []
        current_day = ""
        _nt = []
        _nd = []
        _nn = []
        l += [("", "", "")]  # 可以让最后一天在循环中判断
        for _t, _m, _n in l:
            _cd = get_time_day(_t)
            if _cd != current_day:
                # 判断之前数据的是否较为完整
                if remove_complete_day:
                    if len(_nt) > size * complete_ratio:
                        nt += _nt
                        nd += _nd
                        nn += _nn
                        # print "stay {}, {}/{}".format(current_day,len(_nt),size)
                    else:
                        print "remove {}, just {}/{}".format(current_day, len(_nt), size)
                else:  # 不用去除
                    nt += _nt
                    nd += _nd
                    nn += _nn
                _nt = [_t]
                _nd = [_m]
                _nn = [_n]
                current_day = _cd
            else:
                _nd.append(_m)
                _nt.append(_t)
                _nn.append(_n)

        print "preprocess finish"

        if cache:
            cPickle.dump(nt, open(time_cache_path, "w"))
            cPickle.dump(nd, open(data_cache_path, "w"))
            cPickle.dump(nn, open(trajnum_cache_path, "w"))
            print "cache finish"

    return nt, nd, nn


def fill_by_time(_d):
    _n = _d.shape[0]
    start_index = 0
    _c = 0
    while _c < _n:
        fill_num = 0
        if _d[_c] == 0:  # 需要填补
            _c += 1
            fill_num += 1
            start_index = start_index - 1
            if start_index == -1:
                start_index = _n - 1
                while (_d[start_index] == 0):
                    fill_num += 1
                    start_index -= 1
                while _d[_c] == 0:
                    fill_num += 1
                    _c += 1

                _x = (_d[_c] - _d[start_index]) / float(fill_num + 1)
                _k = 1
                for _z in range(start_index + 1, _n):
                    _d[_z] = _d[start_index] + (_k * _x)
                    _k += 1
                for _z in range(_c):
                    _d[_z] = _d[start_index] + (_k * _x)
                    _k += 1
            else:
                while _c < _n and _d[_c] == 0:
                    fill_num += 1
                    _c += 1
                _x = (_d[_c % _n] - _d[start_index]) / (fill_num + 1)
                _k = 1
                for _z in range(start_index + 1, _c):
                    _d[_z] = _d[start_index] + _k * _x
                    _k += 1

            start_index = _c
        else:
            _c += 1
            start_index = _c
    return _d


def fill_by_road(d, _other):
    _n = d.shape[0]
    for _i in range(_n):
        _o = _other[:, _i]
        _ds = []
        for _d in _o:
            if _d != 0:
                _ds.append(_d)
        if len(_ds) != 0:
            _v = np.mean(_ds)
        else:
            _v = 0
        d[_i] = _v


def get_adjacent_edge_ids(rg, _choose_edge_id, start=True, end=True):
    edge = rg[rg.edge_id == _choose_edge_id]
    s_id = edge["s_id"].values[0]
    e_id = edge["e_id"].values[0]
    adjacent_edge_ids = None
    if start and end:
        adjacent_edge_ids = rg[(rg.e_id == s_id) | (rg.s_id == e_id)]["edge_id"].values
    elif start:
        adjacent_edge_ids = rg[(rg.e_id == s_id)]["edge_id"].values
    elif end:
        adjacent_edge_ids = rg[(rg.s_id == e_id)]["edge_id"].values

    return adjacent_edge_ids


@performance
def completion_data(path, suffix, cache=True,
                    start_hour=8, end_hour=22,
                    time_fill_split=0.5, road_fill_split=0.2,
                    stride_sparse=False, stride_edges=1,
                    A=-1):
    if A != -1:
        fix_A = True
    else:
        fix_A = False
        A = 0

    stm_path = os.path.join(path,
                            CACHE,
                            "stm_{}_{}_{}_{}_{}{}{}".format(suffix, start_hour, end_hour,
                                                            time_fill_split, road_fill_split,
                                                            "_" + str(stride_edges) if stride_sparse else "",
                                                            "_" + str(A) if fix_A else ""))
    arm_path = os.path.join(path,
                            CACHE,
                            "arm_{}_{}_{}_{}_{}{}{}".format(suffix, start_hour, end_hour,
                                                            time_fill_split, road_fill_split,
                                                            "_" + str(stride_edges) if stride_sparse else "",
                                                            "_" + str(A) if fix_A else ""))

    time_cache_path = os.path.join(path, CACHE, "time_by_road_{}_{}_{}.pkl".format(suffix, start_hour, end_hour))
    time_window = int(suffix.split("_")[-1])

    if cache and os.path.exists(stm_path + ".npy"):
        print stm_path, " exist"
        print "load cache"
        stm = np.load(stm_path + ".npy")
        arm = np.load(arm_path + ".npy")
        t = cPickle.load(open(time_cache_path, "r"))
    else:
        print "cache not exist"
        print "complete start"
        # 数据补全
        t, d, num = load_raw_link_speed_by_road_in_time(path, suffix, start_hour=start_hour, end_hour=end_hour)
        d = np.vstack(d).T
        d = d.astype(float)
        num = np.vstack(num).T.astype(int)
        num_int = (num >= time_window).astype(int)
        time_num = d.shape[1]

        d = d * num_int

        rg = load_part_RG(path, suffix[:suffix.rindex("_")])
        id2edgeid = get_id2edgeid(rg)
        edgeid2id = get_edgeid2id(rg)
        # 填补空缺
        dr = []
        choose = []
        for _d in d:
            dr.append((_d != 0).sum() / float(time_num))
        dr = np.asarray(dr)
        for _i, _dr in enumerate(dr):
            if _dr >= time_fill_split:  # 这个road的缺失率小于0.3的时候,按time填补
                choose.append(_i)
                _d = d[_i]
                fill_by_time(_d)

        # new dr
        dr = []
        for _d in d:
            dr.append((_d != 0).sum() / float(time_num))
        dr = np.asarray(dr)
        juge_choose = set(choose)
        for _i, _dr in enumerate(dr):
            if _dr >= road_fill_split and _dr < time_fill_split:  # 这个road的缺失率在一定范围内的时候,先road填补再time
                edgeid = id2edgeid[_i]
                adjacent_edge_ids = get_adjacent_edge_ids(rg, edgeid)
                _choose_adjacent = []
                for _edge_id in adjacent_edge_ids:
                    if edgeid2id[_edge_id] in juge_choose:
                        _choose_adjacent.append(edgeid2id[_edge_id])
                if len(_choose_adjacent) > 0:
                    fill_by_road(d[_i], d[_choose_adjacent])
                    _new_dr = (d[_i] != 0).sum() / float(time_num)
                    if _new_dr > time_fill_split and _new_dr < 1:
                        fill_by_time(d[_i])
                    _new_dr = (d[_i] != 0).sum() / float(time_num)
                    if _new_dr == 1:
                        choose.append(_i)
                        juge_choose.add(_i)

        # 构造最后的spatial-temporal matrix 以及 adjacent road matrix
        choose = sorted(list(set(choose)))
        choose_edge_ids = [id2edgeid[_c] for _c in choose]
        edgeid2newid = dict(zip(choose_edge_ids, range(len(choose))))
        newid2edgeid = dict(zip(range(len(choose)), choose_edge_ids))
        choose_edge_ids_set = set(choose_edge_ids)
        choose_edge_ids_adjacent_ids = []
        if not stride_sparse:  # 不跨边
            for _choose_edge_id in choose_edge_ids:
                adjacent_edge_ids = get_adjacent_edge_ids(rg, _choose_edge_id)
                temp = set()
                for _choose_adjacent_edgeid in adjacent_edge_ids:
                    if _choose_adjacent_edgeid in choose_edge_ids_set:
                        temp.add(edgeid2newid[_choose_adjacent_edgeid])
                temp.add(edgeid2newid[_choose_edge_id])
                temp = sorted(list(temp))
                if len(temp) > A:
                    A = len(temp)
                choose_edge_ids_adjacent_ids.append(sorted(temp))
        else:  # 跨边连接信息
            for _choose_edge_id in choose_edge_ids:
                temp = []
                temp.append(edgeid2newid[_choose_edge_id])
                _found_edge_ids = set()
                _found_edge_ids.add(_choose_edge_id)

                # 找邻接边
                def find_adjacent_edge_ids_by_stride(_choose_edge_id,
                                                     temp,
                                                     _found_edge_ids,
                                                     step=0,
                                                     start=True,
                                                     end=True):
                    if step > stride_edges:  # 超过跨越度
                        return

                    if start:
                        start_adjacent_edge_ids = get_adjacent_edge_ids(rg,
                                                                        _choose_edge_id,
                                                                        start=True,
                                                                        end=False)
                        for _choose_adjacent_edgeid in start_adjacent_edge_ids:
                            # 在候选边中
                            if _choose_adjacent_edgeid not in _found_edge_ids:
                                _found_edge_ids.add(_choose_adjacent_edgeid)
                                if _choose_adjacent_edgeid in choose_edge_ids_set:
                                    if edgeid2newid[_choose_adjacent_edgeid] not in temp:
                                        temp.append(edgeid2newid[_choose_adjacent_edgeid])
                                # 不在候选边中,找它的邻接边信息
                                else:
                                    find_adjacent_edge_ids_by_stride(_choose_adjacent_edgeid,
                                                                     temp,
                                                                     _found_edge_ids,
                                                                     step + 1,
                                                                     start=True,
                                                                     end=False)

                    if end:
                        end_adjacent_edge_ids = get_adjacent_edge_ids(rg, _choose_edge_id, start=False, end=True)
                        for _choose_adjacent_edgeid in end_adjacent_edge_ids:
                            # 在候选边中
                            if _choose_adjacent_edgeid not in _found_edge_ids:
                                _found_edge_ids.add(_choose_adjacent_edgeid)
                                if _choose_adjacent_edgeid in choose_edge_ids_set:
                                    if edgeid2newid[_choose_adjacent_edgeid] not in temp:
                                        temp.append(edgeid2newid[_choose_adjacent_edgeid])
                                # 不在候选边中,找它的邻接边信息
                                else:
                                    find_adjacent_edge_ids_by_stride(_choose_adjacent_edgeid,
                                                                     temp,
                                                                     _found_edge_ids,
                                                                     step + 1,
                                                                     start=False,
                                                                     end=True)

                find_adjacent_edge_ids_by_stride(_choose_edge_id, temp, _found_edge_ids)

                current_times = 0
                while fix_A and len(temp) < A and current_times < 3:
                    for _idd in list(temp):
                        find_adjacent_edge_ids_by_stride(newid2edgeid[_idd], temp, _found_edge_ids)
                    current_times += 1

                # temp = sorted(list(temp))

                if not fix_A:
                    if len(temp) > A:
                        A = len(temp)
                else:
                    temp = temp[:A]

                choose_edge_ids_adjacent_ids.append(temp)

        stm = d[choose]  # spatial-temporal matrix
        arm = np.zeros((len(choose), A))  # adjacent road matrix
        arm[:] = len(choose)
        for _index, _temp in enumerate(choose_edge_ids_adjacent_ids):
            for _j, _v in enumerate(_temp):
                arm[_index, _j] = _v
        arm = arm.astype(int)
        print "complete finish"
        if cache:
            print "cache data"
            np.save(stm_path, stm)
            np.save(arm_path, arm)
            print "cache finish"
    return stm, arm, t


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    hours = [int(t[8:10]) for t in timestamps]
    hour_min = np.min(hours)
    hour_max = np.max(hours)
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
        v2 = [0 for _ in range(hour_max - hour_min + 1)]
        v2[hours[i] - hour_min] = 1
        v+=v2
    return np.asarray(ret)

def load_meteorol(timeslots, fname):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['windspeeds'].value
    Weather = f['weathers'].value
    maxTs = f['maxTs'].value
    minTs = f['minTs'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    maxTE = []  # maxTs
    minTE = []

    for slot in timeslots:
        predicted_id = M[int(slot[:8])]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        maxTE.append(maxTs[cur_id])
        minTE.append(minTs[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    maxTE = np.asarray(maxTE)
    minTE = np.asarray(minTE)
    # 0-1 scale
    if WS.max() - WS.min() != 0:
        WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    else:
        WS[:] = 0
    maxTE = 1. * (maxTE - maxTE.min()) / (maxTE.max() - maxTE.min())
    minTE = 1. * (minTE - minTE.min()) / (minTE.max() - minTE.min())
    print("shape: ", WS.shape, WR.shape, maxTE.shape, minTE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], maxTE[:, None], minTE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_holiday(timeslots, fname):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    # print(timeslots[H==1])
    return H[:, None]


if __name__ == '__main__':
    completion_data()