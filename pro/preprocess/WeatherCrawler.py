#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-26, 14:59

@Description:

@Update Date: 17-7-26, 14:59
"""

import urllib
import urllib2
import re
from bs4 import BeautifulSoup
import Paramater
import h5py

def getPage(month):
    """

    :param month: yyyyMM
    :return:
    """
    url = "http://lishi.tianqi.com/beijing/{}.html".format(month)
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    return response.read().decode('gbk')


def parse(content):
    soup = BeautifulSoup(content, 'lxml')
    # print soup.prettify()
    infos = soup.find(class_="tqtongji2")
    list = infos.find_all('ul')
    r = []
    for v in list:
        ahref = v.find('a')
        if ahref is not None:
            lis = v.find_all('li')
            date = str(lis[0].text.replace("-", ""))
            maxT = str(lis[1].text)
            minT = str(lis[2].text)
            weather = str(lis[3].text.encode("utf-8"))
            ws = str(lis[5].text.encode("utf-8"))
            r.append(",".join((date, maxT, minT, weather, ws)))
    return r




def weatherOneHot(weather):
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if (weather.find("晴") > -1):
        w[0] = 1
    if (weather.find("多云") > -1):
        w[1] = 1
    if (weather.find("阴") > -1):
        w[2] = 1
    if (weather.find("雨") > -1):
        if (weather.find("小") > -1):
            w[3] = 1
        if (weather.find("中") > -1):
            w[4] = 1
        if (weather.find("大") > -1):
            w[5] = 1
        if (weather.find("阵") > -1):
            w[6] = 1
        if (weather.find("暴") > -1):
            w[7] = 1
    if (weather.find("雪") > -1):
        w[8] = 1
    if (weather.find("霾") > -1):
        w[9] = 1
    return w


def windToInt(wind):
    r = 0
    if wind == "微风":
        r = 0
    elif wind.find("小于") > -1:
        r = int(wind[wind.find("级") - 1]) - 1
    elif wind.find("级"):
        r = int(wind[wind.find("级") - 1])
    return r


def create_weather_h5():
    import pandas as pd
    import numpy as np
    dataPath = "/home/zhongjianlv/gpumount/TrafficPrediction/JamPredict/data/" #Paramater.DATAPATH
    csv = pd.read_csv(dataPath + "BJ_WEATHER.csv", names=["date", "maxT", "minT", "weather", "windspeed"])
    arr = np.array(csv)
    date = []
    maxTs = []
    minTs = []
    weathers = []
    windspeeds = []
    for v in arr:
        date.append(v[0])
        maxTs.append(v[1])
        minTs.append(v[2])
        weathers.append(weatherOneHot(v[3]))
        windspeeds.append(windToInt(v[4]))



    f = h5py.File(dataPath + "BJ_WEATHER.h5", "w")
    f.create_dataset("date", data=date)
    f.create_dataset("maxTs", data=maxTs)
    f.create_dataset("minTs", data=minTs)
    f.create_dataset("weathers", data=weathers)
    f.create_dataset("windspeeds", data=windspeeds)
    f.flush()
    f.close()

def main():
    with open(Paramater.DATAPATH + "BJ_WEATHER.csv", "w") as f:
        for i in range(1, 13):
            month = "2016" + "%02d" % i
            l = parse(getPage(month))
            for v in l:
                f.write(v + "\n")
        f.flush()

    create_weather_h5()



if __name__ == '__main__':
    # main()
    # f = h5py.File(Paramater.DATAPATH + "BJ_WEATHER.h5", "r")
    # date = f['date'].value
    # maxTs = f['maxTs'].value
    # minTs = f['minTs'].value
    # weathers = f['weathers'].value
    # windspeeds = f['windspeeds'].value
    # print date,type(date),type(date[1])
    # print maxTs
    # print minTs
    # print weathers
    # print windspeeds
    create_weather_h5()