import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import statsmodels.api as sm
from statsmodels.api import VAR

FILE_PATH = '/root/sampleTest/train/'

file_name = FILE_PATH + '000001.SZ.csv'
df = pd.read_csv(file_name, parse_dates=True)

#checking stationarity
# adfuller()

# checking

#creating the train and validation set
train = df[:int(0.8*(len(df)))]
valid = df[int(0.8*(len(df))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))