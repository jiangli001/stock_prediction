import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
import seaborn as sns

FILE_PATH = '/root/sampleTest/train/'

file_name = FILE_PATH + '000001.SZ.csv'
df = pd.read_csv(file_name, parse_dates=True)
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('M')

# check stationarity with Augmented Dickey-Fuller test
# The null hypothesis of the test is that the time series is non-stationary

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
    
        y: timeseries
        lags: how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

# for col in df.columns:
#     tsplot(df[col], lags=60)

# #fit the model
def fit_model(df, max_p, columns=None):
    """Fit a VAR model

    Args:
        df (pandas DataFrame object): input dataframe
        max_p (int): largest number to loop through for p
        columns (list, optional): a list of column names. Defaults to None.

    Returns:
        VAR model
    """
    
    diff = df[columns].diff().dropna()
    # can fit multiple columns 
    forecasting_model = VAR(diff[columns])
    # try multiple values of p
    for p in range(1, max_p):
        result = forecasting_model.fit(p)
        if p == 1 or result.aic > best_aic:
            best_aic = result.aic
            best_result = result
    return best_result

model_result = fit_model(df, 10, ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'TranscationVolume'])
print(model_result.summary())

def plot_results():
    sns.set()
    plt.plot(list(np.arange(1,10,1)), model_result)
    plt.xlabel("Order")
    plt.ylabel("AIC")
    plt.show()

# # make prediction on validation
# prediction = model_fit.forecast(model_fit.y, steps=len(valid))