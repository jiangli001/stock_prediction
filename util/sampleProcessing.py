import numpy as np
from numpy.core.getlimits import _discovered_machar
import pandas as pd
import os
import shutil


'''

'''


class SampleProcessing:
    def __init__(self):
        self.DaysPerRow = 5 #每行含有的天数，因为担心我们没办法用多因子timeseries
        self.PredictDay= 2 #这个决定targetvalue是多少天的涨幅
        self.AvgTurnoverDaysMin = 5 #换手率短期均值
        self.AvgTurnoverDaysMax = 30 #换手率中长期均值，决定了IPO后多少天的数据我们不用
        self.IlliquilityDay = 30 #非流动性风险均值时间
        self.File = "/root/rawDataTest" #RawData目录
    
    def get_illiquility(self,dt):
        '''
        输入rawdt，输出rawdt
        公式 = （日收益率绝对值/换手率），这个值X天的平均
        '''
        daynum = self.IlliquilityDay 
        
        #计算日收益率绝对值
        DayIncrease = abs(np.array(dt.ClosePrice[1:])/np.array(dt.ClosePrice[0:-1])-1)
        DayIncrease = [0] + DayIncrease.tolist() #补齐第一天数据
        
        #Dayincrease/换手率Turnover
        DItoTO = DayIncrease/dt.Turnover*100
        
        #计算均值
        illiquity = [0]*30
        for i in range(daynum+1,len(DItoTO)+1):
            illiquity = illiquity + [np.average(DItoTO[i-daynum:i])]
        
        dt['illiquility'] = illiquity
        
        return dt
        
        
    
    def get_AvgTurnoverMin(self):
        '''
        输入rawdt，输出rawdt
        短期换手率均值
        '''
        return None
    
    def get_AvgTurnoverMax(self):
        '''
        输入rawdt，输出rawdt
        长期换手率均值
        '''
        return None
    
    def get_sampledata(self,dt):
        '''
        剔除均值前的天数
        把数据整理成一行X天
        增加Target Value
        '''
        daynum = max(self.IlliquilityDay,self.AvgTurnoverDaysMax)
        colday = self.DaysPerRow
        pd = self.PredictDay
        
        #计算TargetValue, 名字为GR
        GR = np.array(dt.ClosePrice[pd:])/np.array(dt.ClosePrice[0:-pd])-1
        GR = GR.tolist() + [0]*pd
        
        #把数据改成一行X天
        if colday >1: #colday大于1，一行天数大于1
            for col in dt.columns[4:]:
                for i in range(1,colday):
                    colname = col + str(i) #设置factors名，如OpenPrice3，代表前3天的OpenPrice
                    coldata = [0]*i + dt[col][i:].tolist()
                    dt[col]
        
        
        #把GR加上去
        dt["GrowthRate"] = GR
        
        #把均值天数之前的删除，因为之前是没有均值的；且把没有GR的删掉
        dt = dt[daynum+1:-pd]
        
        return dt
    
    def split_data(self,dt):
        '''
        train data 2019-12-31之前
        valid 2020.1.1-2020.6.30
        test 2020.6.30之后
        '''
        
        dt1 = dt.set_index('Date')
        traindt = dt1[:'2019']
        testdt = dt1['2020-1-1':'2020-6-30']
        valdt = dt1['2020-6-30':]
        
        return traindt, testdt, valdt
        
    
        

if __name__ == "__main__":
    #删除原来文件，建立新文件夹
    shutil.rmtree(os.path.join('/root','sampleTest','train'), ignore_errors=True)
    shutil.rmtree(os.path.join('/root','sampleTest','test'), ignore_errors=True)
    shutil.rmtree(os.path.join('/root','sampleTest','valid'), ignore_errors=True)
    os.makedirs(os.path.join('/root','sampleTest','train'))
    os.makedirs(os.path.join('/root','sampleTest','test'))
    os.makedirs(os.path.join('/root','sampleTest','valid'))
        
    for file in os.listdir("/root/rawDataTest"):
        #读文件
        filepath = '/root/rawDataTest/' + file
        rawdt = pd.read_csv(filepath)
        
        #把Date变成日期格式
        rawdt['Date'] = pd.to_datetime(rawdt['Date'])
        
        #把数据变成float格式
        for col in rawdt.columns[4:]:
            rawdt[col] = pd.to_numeric(rawdt[col],errors='coerce')
        rawdt = rawdt[(rawdt.Turnover != 0)&(rawdt.TranscationVolume != 0)]
        rawdt['Day'] = range(0,len(rawdt))
        
        #如果股票IPO至今时间少于1000天，则抛弃
        if len(rawdt) < 900:
            print("股票{}，因为IPO时间过短被排除".format(file),)
            print("\n","-----------------------------------","\n")
            continue
        
        print("股票{}继续清理".format(file))
        
        #计算好illiquility
        rawdt = SampleProcessing().get_illiquility(rawdt)
        
        #把所有数据清理好
        sampledt = SampleProcessing().get_sampledata(rawdt)
        print(sampledt)
        print("\n","-----------------------------------","\n")
        
        #删除原来文件并保存新文件
        traindt, testdt, valdt = SampleProcessing().split_data(sampledt)
        traindt.to_csv(os.path.join('/root','sampleTest','train', file))
        testdt.to_csv(os.path.join('/root','sampleTest','test', file))
        valdt.to_csv(os.path.join('/root','sampleTest','valid', file))
