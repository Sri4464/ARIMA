# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid import auto_arima
# **********************************************************
# Issues while importing ARIMA
#https://github.com/tgsmith61591/pmdarima/issues/23 (goto this site) or
#conda create -n pmdissue23 --yes --quiet python=3.5 numpy scipy scikit-learn statsmodels
#activate pmdissue23
#pip install pyramid-arima

# This is predicting the Crude Oil price using the ARIMA model

# Reading the Train Dataset
# The dataset from the file Crude_Oil.csv is divided into Training and Testing Dataset
# We will train the model using the train dataset and see the performance of the model agains test dataset.

# Read the train dataset
train = pd.read_csv("Train.csv", sep=',', parse_dates=True)
print(train.head())
# Summary of the training dataset
print(train.info())

# There are missing values in the dataset but for now we are proceeding without any changes to missing values

# Extracting the Month and Year from the Date column
train['Months'] = pd.DatetimeIndex(train['Date']).month
train['Year'] = pd.DatetimeIndex(train['Date']).year

print(train.head())
print(train.info())
# Note that the Months and Year column are alerady in mumeric format and hence there is no change needed for that
# If the same code is written in R, then the months and year needs to be numeric for applying the filters

# We are calculating the time for applying the Loess Filter (This will be used later)
train['Time'] = train['Year'] + train['Months'] / 12
print(train.head())

# Plotting the Graphs of WTI(Cushing) and BRENT (BRT) overtime
train['Date'] = pd.to_datetime(train['Date'])
plt.plot_date(train['Date'], train['WTI'], color='g')
plt.plot_date(train['Date'], train['BRT'], color='orange')
plt.xlabel('WTI/BRT Price in US Dollars')
plt.ylabel('Time (Year / Months)')
plt.legend(loc='upper left')
plt.title('Price Movement of Crude Oil (WTI/BRT)')
plt.show()

# Using the HP filter to smooth the fluctuations and trying
# to make the series stationary
dta = sm.datasets.macrodata.load()
wti_train = pd.DataFrame(train[['Date', 'WTI']])
print(wti_train.dtypes)
wti_cycle, wti_trend = sm.tsa.filters.hpfilter(wti_train.WTI)
train['WTI_cycle'] = wti_cycle
train['WTI_trend'] = wti_trend

brt_train = pd.DataFrame(train[['Date', 'BRT']])
brt_cycle, brt_trend = sm.tsa.filters.hpfilter(brt_train.BRT)
train['BRT_cycle'] = brt_cycle
train['BRT_trend'] = brt_trend

fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=12
fig_size[1]=12

# Showing the HP filter Noise Cycle and Trend for WTI
fig = plt.figure()
plt.title('WTI HP Filter Price / Cycle / Trend')
ax1 = fig.add_subplot(311)
plt.plot(train['Date'], train['WTI'])
ax1.set_ylabel('WTI price')
ax2 = fig.add_subplot(312)
plt.plot(train['Date'], train['WTI_trend'])
ax2.set_ylabel('Trend')
ax3 = fig.add_subplot(313)
plt.plot(train['Date'], train['WTI_cycle'])
ax3.set_ylabel('Cycle')
plt.show()

# Based on the graphs displayed above the HP filter does not display the necessary expected results hence rejecting it

# Trying with Lowess Filter or seasonal decomposition

series = pd.DataFrame(data=train[['Date', 'WTI']])
series.set_index('Date', inplace=True)

print('$$$$$$$$$$$$')
print(series.head())
print(series.dtypes)
result = seasonal_decompose(series, model='multiplicative', freq=1)
result.plot()
plt.show()

# Lowess Filter also does not provide the expected results

# Using ARIMA Model to predict the time series analysis
# it is not always needed to have the HP and lowess filter analysis, but it was necessary in this case to use them

# WTI ARIMA

wti_train['WTI'] = wti_train['WTI'].apply(str)


stepwise_model_wti = auto_arima(wti_train['WTI'], start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
print('WTI AIC')
print(stepwise_model_wti.aic())

# BRT ARIMA
brt_train.dropna(inplace=True) # because ARIMA cannot work on NULL values


stepwise_model_brt = auto_arima(brt_train['BRT'], start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
print('BRT AIC')
print(stepwise_model_brt.aic())

print("Training the model on our datasets")
stepwise_model_wti.fit(wti_train['WTI'])
stepwise_model_brt.fit(brt_train['BRT'])

print("Forecasting WTI for next 13 periods")
future_forecast_wti = stepwise_model_wti.predict(n_periods=13)
print(future_forecast_wti)

print("Forecasting BRT for next 13 periods")
future_forecast_brt = stepwise_model_brt.predict(n_periods=13)
print(future_forecast_brt)


# Changes for WTI forecasting
wti_train.set_index(wti_train['Date'],inplace=True)
wti_train = wti_train.drop(columns=['Date'])
dts = ['16-May', '16-Jun', '16-Jul', '16-Aug', '16-Sep', '16-Oct', '16-Nov', '16-Dec', '17-Jan', '17-Feb', '17-Mar',
       '17-Apr', '17-May']
future_forecast_wti = pd.DataFrame(future_forecast_wti,columns=['WTI'], index=dts)
new_wti = pd.concat([wti_train,future_forecast_wti])
new_wti.index.name = 'Date'
print(new_wti.head())
plt.plot(new_wti)
plt.xlabel('WTI Price in US Dollars')
plt.ylabel('Time (Year / Months)')
plt.legend(loc='upper left')
plt.title('Price Prediction of WTI')
plt.show()

# The same thing can be done for BRT as well.