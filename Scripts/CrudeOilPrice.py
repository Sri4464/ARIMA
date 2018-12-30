# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid import auto_arima

"""This is predicting the Crude Oil price using the ARIMA model
Reading the Train Dataset
The dataset from the file Crude_Oil.csv is divided into Training and Testing Dataset
We will train the model using the train dataset and see the performance of the model agains test dataset."""

#Read Dataset
train = pd.read_csv("Train.csv", sep=',', parse_dates=True)

# Summary of the training dataset
print(train.info())

# Extracting the Month and Year from the Date column
train['Months'] = pd.DatetimeIndex(train['Date']).month
train['Year'] = pd.DatetimeIndex(train['Date']).year

# We are calculating the time for applying the Loess Filter (This will be used later)
train['Time'] = train['Year'] + train['Months'] / 12

# Plotting the Graphs of WTI(Cushing)
train['Date'] = pd.to_datetime(train['Date'])
plt.plot(train['Date'], train['WTI'], color='g', label='WTI')
plt.plot(train['Date'], train['BRT'], color='orange', label='BRT')
plt.xlabel('Time (Year / Months)')
plt.ylabel('WTI/BRT Price in US Dollars')
plt.legend(loc='upper left')
plt.title('Price Movement of Crude Oil (WTI/BRT)')
plt.show()

# Using the HP filter to smooth the fluctuations and trying
# to make the series stationary
dta = sm.datasets.macrodata.load()
wti_train = pd.DataFrame(train[['Date', 'WTI']])

#Extracting the Cycle and Trend using the HP Filter for WTI prices
wti_cycle, wti_trend = sm.tsa.filters.hpfilter(wti_train.WTI)
train['WTI_cycle'] = wti_cycle
train['WTI_trend'] = wti_trend

#Extracting the Cycle and Trend using the HP Filter for BRT prices
brt_train = pd.DataFrame(train[['Date', 'BRT']])
brt_cycle, brt_trend = sm.tsa.filters.hpfilter(brt_train.BRT)
train['BRT_cycle'] = brt_cycle
train['BRT_trend'] = brt_trend

#Plotting the figures

plt.subplot(3, 1, 1)
plt.title('WTI Cycle and Trend Using HP Filter')
plt.plot(train['Date'], train['WTI'], 'k')
plt.ylabel('WTI')
plt.subplot(3, 1, 2)
plt.plot(train['Date'], train['WTI_cycle'], 'k')
plt.ylabel('Cycle')
plt.subplot(3, 1, 3)
plt.plot(train['Date'], train['WTI_trend'],'k')
plt.ylabel('Trend')
plt.xlabel('Time (Years)')
plt.tight_layout()
plt.show()

#Clearing the screen of clutter
clear = "\n" * 100
print(clear)

# Based on the graphs displayed above the HP filter does not display the necessary expected results hence rejecting it
# Trying with Lowess Filter or seasonal decomposition
series = pd.DataFrame(data=train[['Date', 'WTI']])
series.set_index('Date', inplace=True)
print(series.head())
print(series.dtypes)
result = seasonal_decompose(series, model='multiplicative', freq=1)
result.plot()
plt.show()

# Lowess Filter also does not provide the expected results
# Using ARIMA Model to predict the time series analysis
# it is not always needed to have the HP and lowess filter analysis, but it was necessary in this case to use them
# WTI ARIMA
print(clear)

stepwise_model_wti = auto_arima(wti_train['WTI'], start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
print('WTI AIC')
print(stepwise_model_wti.aic())

print("Training the model on our datasets")
stepwise_model_wti.fit(wti_train['WTI'])

print("Forecasting WTI for next 13 periods")
future_forecast_wti = stepwise_model_wti.predict(n_periods=13)
dts = ['16-May', '16-Jun', '16-Jul', '16-Aug', '16-Sep', '16-Oct', '16-Nov', '16-Dec', '17-Jan', '17-Feb', '17-Mar',
       '17-Apr', '17-May']
wti_forecasted = pd.DataFrame(future_forecast_wti,columns=['WTI'], index=dts)
wti_forecasted.index.name = 'Date'
print(wti_forecasted)


""" This is the actual data (Actuals vs Predicted
Actuals		Predicted	
Date	WTI	Date	WTI
16-May 	46.71	16-May	41.532237
16-Jun	48.76	16-Jun	42.985654
16-Jul	44.65	16-Jul	43.243673
16-Aug	44.72	16-Aug	42.123423
16-Sep	45.18	16-Sep	41.211393
16-Oct	49.78	16-Oct	40.96679
16-Nov	45.66	16-Nov	41.030223
16-Dec	51.97	16-Dec	40.64595
17-Jan	52.5	17-Jan	42.890221
17-Feb	53.47	17-Feb	42.471189
17-Mar	49.33	17-Mar	48.274536
17-Apr	51.06	17-Apr	49.869228
17-May	48.48	17-May	48.983116
"""

# Issues while importing ARIMA
#https://github.com/tgsmith61591/pmdarima/issues/23 (goto this site) or
#conda create -n pmdissue23 --yes --quiet python=3.5 numpy scipy scikit-learn statsmodels
#activate pmdissue23
#pip install pyramid-arima