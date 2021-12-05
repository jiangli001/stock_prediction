import pandas as pd
import numpy as np

def l2_normalize(x, axis=0, keepdims=True):
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=keepdims)
    return x / norm

def min_max_scale(x):
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def get_data(path="", input_strategy=1):
    """根据 input_strategy 获得数据
    """
    X_cols_name = ["OpenPrice", "ClosePrice", "HighPrice", "LowPrice", "TranscationVolume",
                 "TranscationValue", "Turnover", "WTurnover", "Mturnover", "PE", "PS",
                 "IncreaseDays", "DropdDays", "DMA", "EMA", "MTM", "SAR", "B3612", "BIAS", "MFI",
                 "PWMI", "StrongPeriod", "Bottom", "beta", "STD", "illiquility", "OpenPriceIndex",
                 "ClosePriceIndex", "HighPriceIndex", "LowPriceIndex", "TranscationVolumeIndex",
                 "TranscationValueIndex", "TurnoverIndex"]
    y_col_name = ["GrowthRate"]
    # X_cols_name = ["OpenPrice", "HighPrice", "LowPrice", "TranscationVolume",
    #              "TranscationValue", "Turnover", "WTurnover", "Mturnover", "PE", "PS",
    #              "IncreaseDays", "DropdDays", "DMA", "EMA", "MTM", "SAR", "B3612", "BIAS", "MFI",
    #              "PWMI", "StrongPeriod", "Bottom", "beta", "STD", "illiquility", "OpenPriceIndex",
    #              "ClosePriceIndex", "HighPriceIndex", "LowPriceIndex", "TranscationVolumeIndex",
    #              "TranscationValueIndex", "TurnoverIndex"]
    # y_col_name = ["ClosePrice"]
    
    data = pd.read_csv(path)
    X, y = data[X_cols_name].values, data[y_col_name].values
    
    if input_strategy == 1:
        # X = l2_normalize(X, axis=0)
        X = min_max_scale(X)
        X = np.expand_dims(X, axis=0) # [1, samlples, 33]
        y = np.multiply(np.reshape(y, [1, -1, 1]), 1)  # [1, samples, 1]
            
    elif input_strategy == 2:
        return 

    elif input_strategy == 3:
        # X = l2_normalize(X, axis=0)
        X = min_max_scale(X)
        X = np.expand_dims(X, axis=0) # [1, samlples, 33]
        y = np.multiply(np.reshape(y, [1, -1, 1]), 1)  # [1, samples, 1]
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


if __name__ == "__main__":
    path = "/root/sampleTest/train/000001.SZ.csv"
    X, y = get_data(path=path, input_strategy=1)
    print(X)