from posix import XATTR_SIZE_MAX
from typing import Sized
import numpy as np
from numpy.core.getlimits import _discovered_machar
import pandas as pd
import os
import shutil
import random
import datetime
import math
import matplotlib.pyplot as plt 
from pandas.io.parsers import read_csv


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)



##################################################
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


#创建文件夹test_Random
shutil.rmtree(os.path.join('/root','sampleTest','test_Random'), ignore_errors=True)
os.makedirs(os.path.join('/root','sampleTest','test_Random'))

#test数据生成随机的一列(用5天平均)，仅用来测试，正常时候不用
for file in os.listdir('/root/sampleTest/test/'):
        #读文件
        filepath = '/root/sampleTest/test/' + file
        dt = pd.read_csv(filepath, index_col= 0)
        # grmax = max(dt['GrowthRate'])
        # grmin = min(dt['GrowthRate'])
        predict_day = 5
        GrowthRateRandom = [0]*(predict_day+2)
        for i in range(predict_day+2,len(dt)):
            gr = np.average(dt['GrowthRate'].values[(i-predict_day-2):(i-2)])
            GrowthRateRandom = GrowthRateRandom + [gr]
        dt['GrowthRatePredict'] = GrowthRateRandom
        dt.to_csv(os.path.join('/root','sampleTest','test_Random', file))

########################################结束


'''
以下为回测逻辑
模型更新：需要在模型模块完成
回测模块主要涉及如下模块：
买卖决策以及交易记录、总资产情况，TradeRecord：dataframe记录，Date、Buy(该数据格式为字典，股票、多少手、现价、金额，如{stocks:[],Hands:[],Value:[]})，sell（同buy）、
        HoldingShare（同buy）、TotalAsset、InvestedAsset、RemainingAsset、Return（相比于第一天的涨跌幅）

每一条记录，是指当天收盘后的情况

买卖策略：

第n天：（收盘工作）
先选择GrowthRatePredict的前 TopPredictNum 名（就是前几名，但这个几在后续用TopPredictNum来定义，如TopPredictNum = 10则是前10）
TopPredict的股票需要预测涨幅需要大于 BuyThreshold
同时满足以上两个条件的，选择成为WaitList
最后WaitList要加上自己的持仓

第n+1天：（快收盘工作）
在WaitList里面，在做一次predict，选择出前 HoldNum 名的股票（选出持仓数量的股票），且n+1天的GrowthRatePredict需要大于BuyThreshold
对比目前的持仓股票：
    如果相同：维持持仓
    如果有不同：调整持仓

买入仓位分配需要设置好
'''




# d = (pd.DataFrame({'A':[[1,2,3]]}).append(pd.DataFrame({'A':[[1,2,3],[22]]})))
# d.reset_index(drop=True)
# dict(zip(['1','2','3'],['a','b','c']))
# StartDate-datetime.timedelta(days=1)
# datetime.datetime.strptime('2020-01-03', "%Y-%m-%d").strftime('%Y-%m-%d')



class TradeStrategy():
    
    def __init__(self):
        self.StartDate = '2020-06-30'   #开始日期
        self.EndDate = '2021-09-30' #结束日期
        self.Initiation = 150000.00 #初始投入
        self.BuyThreshold = 0.02 #买入门槛，两天涨幅预测低于这个数字不买
        self.TopPredictNum = 50 #waitlist股票数量
        self.HoldNum = 6 #持仓数量
        self.FilePath = os.path.join('/root','sampleTest','test_Random') #预测好的test数据存放文件夹
        self.Hands = 100 #一手多少股
        self.Fee = 0.003 #佣金费率
        self.Fee_min = 5 #最低佣金
        self.FeePrepare = 500 #预留一部分钱支付手续费
        self.Harvest = True #每次投资只用初始投资的钱，多余的钱留着空仓
    
   
    def get_all_GrowthRate(self, date, stocks):
        
        '''
        输出dataframe，里面有规定日期的Stock，GrowthRatePredict, ClosePrice
        按照GrowthRatePredict排序
        '''
        AllData = pd.DataFrame({'Stock':[],'GrowthRatePredict':[],'ClosePrice':[]})
        
        for file in stocks:
            dt = pd.read_csv(os.path.join(self.FilePath, file), index_col= 0)
            try:
                GrowthRatePredict = dt[date:date]['GrowthRatePredict'].values[0]
                ClosePrice = dt[date:date]['ClosePrice'].values[0]
                newdt = pd.DataFrame({'Stock': [file], 'GrowthRatePredict':[GrowthRatePredict],'ClosePrice':[ClosePrice]})
                AllData = AllData.append(newdt)
                
            except:
                continue #如果错误，说明没有这个日期，表示当天不开盘，或者股票停牌
            
        #如果全空，则返回none
        if len(AllData) == 0:
            # print('{} is Not a trading day'.format(date))
            return None
        else:
            AllData_sorted = AllData.sort_values(by = 'GrowthRatePredict', ascending = False)
            return AllData_sorted
    
    
    def get_waitlist(self, date):
        '''
        获得Toppredictnum数量的股票
        '''
        stocks = os.listdir(self.FilePath)
        Alldt = self.get_all_GrowthRate(date, stocks)
                
        #如果全空，则返回none
        if Alldt is None:
            return None
        else:
            return Alldt.head(self.TopPredictNum)  


    def get_holdingshare(self, date, totalasset, stop_trading = None):
        '''
        获得当天决定后的持仓，需要输入当前的总金额
        输出HoldingShare（字典，股票、多少手、现价、金额）
        '''
        #如果是第0天，则持股为0
        if date == self.StartDate:
           HoldingShare = None
           
        else:
            #拿到前一天的waitlist，考虑到date前一天可能出现非交易日，因此用while循环
            waitlist1 = None
            date0 = datetime.datetime.strptime(date, "%Y-%m-%d")
            while waitlist1 is None:
                date0 = date0 - datetime.timedelta(days=1)
                waitlist1 = self.get_waitlist(date0.strftime('%Y-%m-%d'))
                
            #拿到waitlist里面，当天的GrowthRatePredict
            newpredict = self.get_all_GrowthRate(date, waitlist1['Stock'].tolist()).head(self.HoldNum)
            
            #判断是否大于买入门槛 
            Holdinglist = newpredict[newpredict['GrowthRatePredict'] > self.BuyThreshold]
            
            
            #如果出都小于门槛，则不买
            if len(Holdinglist) == 0:
                    HoldingShare = stop_trading
            else:
                if self.Harvest:
                    money = (min(self.Initiation, totalasset) - self.FeePrepare)/len(Holdinglist)
                else:
                    money = (totalasset - self.FeePrepare)/len(Holdinglist) #留一部分钱用来付佣金,剩下的钱均分到需要买的股票
                stock_remove = []
                price_remove = []
                if not stop_trading is None:#如果有停牌股票
                    money = money - sum(stop_trading['Value'])
                    stop_num = len(stop_trading['Stock'])
                    stocks = stop_trading['Stock'] + Holdinglist['Stock'].tolist()[0: (self.HoldNum-stop_num)]
                    prices = stop_trading['Price'] + Holdinglist['ClosePrice'].tolist()[0: (self.HoldNum - stop_num)]
                    HoldingHands = stop_trading['Hands']
                    HoldingValues = stop_trading['Value']
                    for stock, price in zip(stocks[stop_num:], prices[stop_num:]):
                        HoldingHand = math.floor(money/(price*self.Hands))
                        if HoldingHand != 0:
                            HoldingValue = self.Hands*price*HoldingHand
                            HoldingHands = HoldingHands + [HoldingHand]
                            HoldingValues = HoldingValues + [HoldingValue]
                        else:
                            stock_remove = stock_remove + [stock]
                            price_remove = price_remove + [price]
                    
                else:
                    stocks = Holdinglist['Stock'].tolist()
                    prices = Holdinglist['ClosePrice'].tolist()
                    HoldingHands = []
                    HoldingValues = []
                    for stock, price in zip(stocks, prices):
                        HoldingHand = math.floor(money/(price*self.Hands))
                        if HoldingHand != 0:
                            HoldingValue = self.Hands*price*HoldingHand
                            HoldingHands = HoldingHands + [HoldingHand]
                            HoldingValues = HoldingValues + [HoldingValue]
                        else:
                            stock_remove = stock_remove + [stock]
                            price_remove = price_remove + [price]
                
                try:
                    for stock, price in zip(stock_remove, price_remove):
                        stocks.remove(stock)
                        prices.remove(price)
                except:
                    pass
                        
                HoldingShare = {'Stock': stocks, 'Hands': HoldingHands, 'Price': prices, 'Value':HoldingValues}
                    
        return HoldingShare    
    
    
    def get_trading_fee(self, decision):
        trading_fee = sum(max(self.Fee_min, x * self.Fee)
                          for x in decision['Value'])
        
        return trading_fee
    
    
    def get_trade_record_day0(self):
        TradeRecord = pd.DataFrame({"Date": [self.StartDate], 
                                    "Buy":[None],
                                    "Sell":[None],
                                    "HoldingShare": [None],
                                    "TotalAsset":[self.Initiation],
                                    "InvestedAsset": [0],
                                    "RemainingAsset":[self.Initiation],
                                    "Return":[0]})
        
        TradeRecord['Date'] = pd.to_datetime(TradeRecord['Date'])
        
        return TradeRecord.set_index('Date')


    def get_Trade_Record(self, date, traderecord):
        '''
        输出TradeRecord数据表
        '''
        #先找到前一个交易日
        day_before_date = datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=1)
        wl = self.get_waitlist(day_before_date.strftime("%Y-%m-%d"))
        while wl is None:
            day_before_date = day_before_date - datetime.timedelta(days=1)
            wl = self.get_waitlist(day_before_date.strftime("%Y-%m-%d"))
            
        day_before_date = day_before_date.strftime("%Y-%m-%d")
        
        
        #拿到前一个交易日的HoldingShare
        total_asset_day_before = traderecord[day_before_date:day_before_date]['TotalAsset'].values[0]
        HoldingShare_day_before = traderecord[day_before_date:day_before_date]['HoldingShare'].values[0]
        if not HoldingShare_day_before is None:
            HoldingShare_day_before = eval(str(HoldingShare_day_before))
        # HoldingShare_day_before = self.get_holdingshare(day_before_date, total_asset_day_before)
        
        # 拿到交易前的total asset
        # 如果前一个交易日没有买股票，那total asset没什么变化
        stop_trading_list = None
        if HoldingShare_day_before is None:
            total_asset_before_trading = total_asset_day_before
        else:
            #如果前一个交易日买了股票，total asset befreo trading就是股票现值加上没有投出去的钱
            #先拿到持股的股票和持股数量
            holdings = HoldingShare_day_before['Stock']
            num_of_holdings = HoldingShare_day_before['Hands']
            
            #再拿到这些股票的当天收盘价
            dtframe = self.get_all_GrowthRate(date, holdings)
            prices = []
            stop_trading_holdings = []
            stop_trading_hands = []
            stop_trading_prices = []
            for stock, hand, price in zip(holdings, num_of_holdings, HoldingShare_day_before['Price']):
                try:
                    p = dtframe[dtframe['Stock']==stock]['ClosePrice'].values[0]
                    prices = prices + [p]
                except:#如果拿不到p代表这只股票停牌了
                    p = price
                    prices = prices + [p]
                    stop_trading_holdings = stop_trading_holdings + [stock]
                    stop_trading_hands = stop_trading_hands + [hand]
                    stop_trading_prices = stop_trading_prices + [price]
                    values = [x * y * self.Hands for x, y in zip(stop_trading_hands, stop_trading_prices)]
                    stop_trading_list = {'Stock':stop_trading_holdings, 'Hands': stop_trading_hands, 'Price': stop_trading_prices, 'Value':values}
            
            #再拿到持仓金额
            holdings_value = sum(x * y * self.Hands for x, y in zip(prices, num_of_holdings))
            #再拿到没投资的钱
            remaining = traderecord[day_before_date:day_before_date]['RemainingAsset'].values[0]
            #加起来
            total_asset_before_trading = remaining + holdings_value
        
        # 拿到现在的HoldingShare
        HoldingShare = self.get_holdingshare(date, total_asset_before_trading, stop_trading = stop_trading_list)
        
        
        #拿到Buy、Sell decision、和total asset
        try:
            a = (set(HoldingShare['Stock']) == set(HoldingShare_day_before['Stock']))
            b = (set(HoldingShare['Hands']) == set(HoldingShare_day_before['Hands']))
        except:
            a = False
            b = False
        if HoldingShare == HoldingShare_day_before or (a & b):
            Buy = None
            Sell = None
            RemainingAsset = traderecord[day_before_date:day_before_date]['RemainingAsset'].values[0]
            if HoldingShare_day_before is None:
                InvestedAsset = 0
            else:
                InvestedAsset = holdings_value
            TotalAsset = RemainingAsset + InvestedAsset
        else:
            if HoldingShare is None: #如果今天HoldingShare是None，则卖掉所有手上股票
                Buy = None
                Sell = HoldingShare_day_before
                Sell['Price'] = prices #改到最新价格
                Sell['Value'] = [x * y * self.Hands for x, y in zip(prices, num_of_holdings)] #改到最新价格
                #拿到交易的价格
                trade_fee = self.get_trading_fee(Sell)
                TotalAsset = sum(Sell['Value']) - trade_fee + traderecord[day_before_date:day_before_date]['RemainingAsset'].values[0]
                InvestedAsset = 0
                RemainingAsset = TotalAsset
            
            elif HoldingShare_day_before is None: #如果昨天的HoldingShare是None，则买所有的
                Buy = HoldingShare
                Sell = None
                trade_fee = self.get_trading_fee(Buy)
                InvestedAsset = sum(Buy['Value'])
                RemainingAsset = traderecord[day_before_date:day_before_date]['RemainingAsset'].values[0] - InvestedAsset - trade_fee
                TotalAsset = RemainingAsset + InvestedAsset
            
            else:
                #先判断买什么
                buyhands = []
                buystocks = []
                buyprices = []
                buyvalue = []
                for stock, hand, price in zip(HoldingShare['Stock'], HoldingShare['Hands'], HoldingShare['Price']):
                    if stock in HoldingShare_day_before['Stock']:
                        order = HoldingShare_day_before['Stock'].index(stock)
                        holdhand = HoldingShare_day_before['Hands'][order]
                        if hand > holdhand:
                            buyhand = hand - holdhand
                            buyhands = buyhands + [buyhand]
                            buystocks = buystocks + [stock]
                            buyprices = buyprices + [price]
                            buyvalue = buyvalue + [buyhand * self.Hands * price]
                            
                    else:
                        buyhand = hand
                        buyhands = buyhands + [buyhand]
                        buystocks = buystocks + [stock]
                        buyprices = buyprices + [price]
                        buyvalue = buyvalue + [buyhand * self.Hands * price]
                            
                Buy = {'Stock': buystocks, 'Hands': buyhands, 'Price': buyprices, 'Value': buyvalue}
                
                #再判断卖什么
                sellhands = []
                sellstocks = []
                sellprices = []
                sellvalue = []
                for stock, hand, price in zip(HoldingShare_day_before['Stock'], HoldingShare_day_before['Hands'], prices):
                    if stock in HoldingShare['Stock']:
                        order = HoldingShare['Stock'].index(stock)
                        buyhand = HoldingShare['Hands'][order]
                        if hand > buyhand:
                            sellhand = hand - buyhand
                            sellhands = sellhands + [sellhand]
                            sellstocks = sellstocks + [stock]
                            sellprices = sellprices + [price]
                            sellvalue = sellvalue + [sellhand * self.Hands * price]
                            
                    else:
                        sellhand = hand
                        sellhands = sellhands + [sellhand]
                        sellstocks = sellstocks + [stock]
                        sellprices = sellprices + [price]
                        sellvalue = sellvalue + [sellhand * self.Hands * price]
                            
                Sell = {'Stock': sellstocks, 'Hands': sellhands, 'Price': sellprices, 'Value': sellvalue}
                trade_fee_buy = self.get_trading_fee(Buy) 
                trade_fee_sell = self.get_trading_fee(Sell)
                
                try:
                    if len(Buy['Stock']) == 0:
                        Buy = None
                        trade_fee_buy = 0
                
                except:
                    pass
                
                try:
                    if len(Sell['Stock']) == 0:
                        Sell = None
                        trade_fee_sell = 0
                except:
                    pass
                
                trade_fee = trade_fee_buy + trade_fee_sell
                InvestedAsset = sum(HoldingShare['Value'])
                TotalAsset = total_asset_before_trading - trade_fee
                RemainingAsset = TotalAsset - InvestedAsset
                
        #拿到return
        Return = TotalAsset/traderecord['TotalAsset'].values[0] - 1
        
        
        #合并表格
        traderecord_update = pd.DataFrame({"Date": [date], 
                                    "Buy":[Buy],
                                    "Sell":[Sell],
                                    "HoldingShare": [HoldingShare],
                                    "TotalAsset":TotalAsset,
                                    "InvestedAsset": InvestedAsset,
                                    "RemainingAsset":RemainingAsset,
                                    "Return":Return})
        
        traderecord_update['Date'] = pd.to_datetime(traderecord_update['Date'])
        traderecord = traderecord.append(traderecord_update.set_index('Date'))

        return traderecord
    
    
    def get_all_TradeRecord(self):
        TradeRecord = self.get_trade_record_day0()
        date = datetime.datetime.strptime(self.StartDate, "%Y-%m-%d")
        enddate = datetime.datetime.strptime(self.EndDate, "%Y-%m-%d")
        while date < enddate:
            date = date + datetime.timedelta(days=1)
            date_str = date.strftime('%Y-%m-%d')
            tradeday = self.get_waitlist(date_str)
            if not tradeday is None:
                try:
                    TradeRecord = self.get_Trade_Record(date_str , TradeRecord)
                
                except:
                    print("{} is abnormal".format(date_str))
                    hs = TradeRecord['HoldingShare'].values[-1]
                    ta = TradeRecord['TotalAsset'].values[-1]
                    ia = TradeRecord['InvestedAsset'].values[-1]
                    ra = TradeRecord['RemainingAsset'].values[-1]
                    re = TradeRecord['Return'].values[-1]
                    TradeRecord_update = pd.DataFrame({"Date": date_str, 
                                    "Buy":[None],
                                    "Sell":[None],
                                    "HoldingShare": hs,
                                    "TotalAsset": ta,
                                    "InvestedAsset": ia,
                                    "RemainingAsset":ra,
                                    "Return":re})
                    
                    TradeRecord_update['Date'] = pd.to_datetime(TradeRecord_update['Date'])
                    TradeRecord = TradeRecord.append(TradeRecord_update.set_index('Date'))        
        return TradeRecord
        
        


if __name__ == "__main__":
    path = os.path.join('/root','sampleTest','test_Random_tradeRecord', 'TradeRecord.csv') #保存trade_record的位置
    TradeRecord = TradeStrategy().get_all_TradeRecord()
    TradeRecord.to_csv(path)    