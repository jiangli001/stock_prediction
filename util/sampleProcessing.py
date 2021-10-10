import numpy as np
import pandas as pd

def sample_processing(stock_id, win_size=2): # TODO: 参数和注释可以按具体情况完善
    """样本处理
    Inputs:
        stock_id [String or Int]: 股票代码
        win_size [Int]: 默认2。预测窗口
    """
    load_path = "/root/rawData/{}.csv".format(stock_id)
    # save_path = "/root/sample/{}_{}.csv".format(stock_id, win_size)
    # TODO: 待完善
    
    return 


def missing_values_cleaning():
    return


def outliers_cleaning():
    return


def data_scaler():
    return 


if __name__ == "__main__":
    print(1)
