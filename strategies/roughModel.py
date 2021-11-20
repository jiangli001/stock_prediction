import numpy as np
from numpy.core.getlimits import _discovered_machar
import pandas as pd
import os
import shutil
import random
import datetime
import math
import matplotlib.pyplot as plt 


'''
以下为建立一个用来测试的数据代码，后续需要删除
'''

#打印设置
##显示所有列
pd.set_option('display.max_columns', None)
##显示所有行
pd.set_option('display.max_rows', None)
##设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)



def moving_average(predict_day, dt):
    GrowthRateRandom = [0]*(predict_day+2)
    for i in range(predict_day+2,len(dt)):
        gr = np.average(dt['GrowthRate'].values[(i-predict_day-2):(i-2)])
        GrowthRateRandom = GrowthRateRandom + [gr]
    return GrowthRateRandom


def TurnoverRate_over_WTurnoverRate(dt):
    GrowthRateRandom = dt['Turnover']/dt['WTurnover']-1
    return GrowthRateRandom



#创建文件夹test_Random
shutil.rmtree(os.path.join('/root','sampleTest','test_Random'), ignore_errors=True)
os.makedirs(os.path.join('/root','sampleTest','test_Random'))

#test数据生成随机的一列(用5天平均)，仅用来测试，正常时候不用
for file in os.listdir('/root/sampleTest/test/'):
        #读文件
        filepath = '/root/sampleTest/test/' + file
        dt = pd.read_csv(filepath, index_col= 0)
        dt['GrowthRatePredict'] = moving_average(5, dt)
        # dt['GrowthRatePredict'] = TurnoverRate_over_WTurnoverRate(dt)
        dt.to_csv(os.path.join('/root','sampleTest','test_Random', file))




