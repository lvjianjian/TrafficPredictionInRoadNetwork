#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-23, 15:11

@Description:

@Update Date: 17-11-23, 15:11
"""

import yaml
import os


class Config(object):
    def __init__(self, config_path):
        self.config_name = os.path.basename(config_path).split(".")[0]
        f = open(config_path, "r")
        conf = yaml.load(f)
        self.data_path = conf["data_path"]
        self.suffix = conf["suffix"]
        self.observe_length = conf["observe_length"]
        self.predict_length = conf["predict_length"]
        self.test_ratio = conf["test_ratio"]
        self.model_path = conf["model_path"]
        self.time_window = conf["time_window"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.early_stopping = conf["early_stopping"]
        self.learning_rate = conf["learning_rate"]
        self.use_cache_model = conf["use_cache_model"]
        self.time_fill_split = conf["time_fill_split"]
        self.road_fill_split = conf["road_fill_split"]
        self.stride_sparse = conf["stride_sparse"]
        self.stride_edges = conf["stride_edges"]
        if not self.stride_sparse:
            self.stride_edges = 0
        self.model_name = conf["model_name"]
        if "L" in self.model_name:
            self.use_loopup = True
        else:
            self.use_loopup = False
        self.no_adjacent_fill_zero = conf["no_adjacent_fill_zero"]
        self.fix_adjacent_road_num = conf["fix_adjacent_road_num"]
