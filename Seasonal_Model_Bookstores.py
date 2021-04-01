import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# augmented dickey fuller
from statsmodels.tsa.stattools import adfuller
# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Import this package for test cointegration
from statsmodels.tsa.stattools import coint
# seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# seasonal modeling
import pmdarima as pm
# SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Change working directory to save graphs
os.getcwd()
os.chdir('C:\\Users\\brand\\PycharmProjects\\Big Data Analytics\\FinalProject')

# use the api to pull in the data from FRED
# api_key=329961d88c12ef9d91df6cc8e64c6900
fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

# get the data for E-Commerce Retail Sales
e_commerce = fred.get_series('ECOMNSA')

# get the data for retail sales: book stores
book_stores = fred.get_series('MRTSSM451211USN')
book_stores.plot()
plt.axvline(x='2007-01-01', c='red', ls='--')
plt.title('Book Store Retail Sales\nRed Line = Introduction of Kindle(2007)')
#plt.savefig('book_store_sales.png')
plt.show()
#book_stores = np.log(book_stores)
#----------------------------------------------------
# DECOMPOSE
#----------------------------------------------------
# To decompose data we need to know how often the cycle repeats
# Use the ACF to identify frequency (or use an educated guess)
# the data need to be de-trended by subtracting the rolling average
#----------------------------------------------------
# BOOK STORE - DETREND - ACF
#----------------------------------------------------
# detrend the data
book_detrend = book_stores - book_stores.rolling(15).mean()
book_detrend = book_detrend.dropna()
# plot the detrended data
# book_detrend.plot()
# plt.title('Book Store Data De-trended')
# plt.show()

# plot the acf
plot_acf(book_detrend, lags=25, zero=False)
plt.title('Book Store ACF')
plt.tight_layout()
#plt.savefig('book_store_acf.png')
plt.show()

# decompose
book_decomp = seasonal_decompose(book_stores, freq=12)
book_decomp.plot()
plt.title('Book Store Seasonal Decomposition')
plt.tight_layout()
#plt.savefig('book_store_decomp.png')
plt.show()

#----------------------------------------------------
# SEASONAL DIFFERENCING
#----------------------------------------------------
# When the time series shows trend, take the difference
# when the time series shows seasonality, take the seasonal difference

# to take the seasonal difference use the following code:
# df_diff = df.diff(S)
# S = length of seasonal cycle

# to find p and q, plot the ACF and PACF
# to find the seasonal order P and Q take ACF and PACF of differenced time series
# at multiple seasonal steps
#----------------------------------------------------
# BOOK STORES - SEASONAL DIFFERENCING
#----------------------------------------------------
# S = 12
# create a new variable w/ seasonal differencing
book_seasonal_diff = book_stores.diff().diff(12).dropna()

# plot it
# book_seasonal_diff.plot(grid=True)
# plt.show()

# ADF
adfuller(book_seasonal_diff)

# plot the acf and pacf for non-seasonality
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,6))
# plot the acf
plot_acf(book_seasonal_diff, lags=11, zero=False, ax=ax1)
# plot the pacf
plot_pacf(book_seasonal_diff, lags=11, zero=False, ax=ax2)
plt.title('ACF and PACF for Book Stores')
plt.tight_layout()
plt.savefig('book_store_non_seasonal_acf_pacf.png')
plt.show()

# plot the acf and pacf for seasonality
lags=[12, 24, 36, 48, 60]
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,6))
# plot the acf
plot_acf(book_seasonal_diff, lags=lags, zero=False, ax=ax1)
# plot the pacf
plot_pacf(book_seasonal_diff, lags=lags, zero=False, ax=ax2)
plt.title('Seasonal ACF and PACF for Book Stores')
plt.tight_layout()
plt.savefig('book_store__seasonal_acf_pacf.png')
plt.show()

#----------------------------------------------------
# ELECTRONIC SHOPPING - SEASONAL MODELING
#----------------------------------------------------
results = pm.auto_arima(book_stores,             # data
                        d=1,            # non-seasonal difference order
                        start_p=1,      # initial guess for p
                        start_1=4,      # initial guess for q
                        max_p=5,        # max vale of p to test
                        max_q=5,        # max value of q to test
                        seasonal=True,  # is the time series seasonal
                        m=12,            # the seasonal period
                        D=1,            # seasonal difference order
                        start_P=0,      # initial guess for P
                        start_Q=0,      # initial guess for Q
                        max_P=5,        # max value of P to test
                        max_Q=5,        # max value of Q to test
                        information_criterion= 'aic',    # used to select best model
                        trace=True,                     # print results whilst training
                        error_action='ignore',          # ignore orders that don't work
                        stepwise=True,                  # apply intelligent order search
                        )
# print the results
print(results.summary())

# plot the diagnostics from the auto arima model
# results.plot_diagnostics()
# plt.tight_layout()
# plt.show()

# ========
# FORECAST
# ========
# create forecast object
model = SARIMAX(book_stores
                , order = (0,1,2)
                , seasonal_order=(1,1,0,12)
                , trend='c'
                )

result = model.fit()

# see the diagnostics
result.plot_diagnostics()
plt.title('Book Store Model Diagnostics')
plt.tight_layout()
#plt.savefig('book_store_model_diag.png')
plt.show()

# create a forecast object
for_obj = result.get_forecast(steps=48)
# extract predicted mean attribute
mean = for_obj.predicted_mean
# confidence intervals
conf_int = for_obj.conf_int()
# forecast dates
dates = mean.index

# PLOT
plt.figure()
# plot past electronic sales
plt.plot(book_stores.index, book_stores, label='past')
# plot the prediction means as line
plt.plot(dates, mean, label='predicted')
# shade between the confidence intervals
plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.7, color='pink')
# plot legend and show fig
plt.legend()
plt.title('Predicted Sales for Book Stores')
plt.axhline(y=0, ls='--', color='red')
plt.tight_layout()
#plt.savefig('book_store_prediction.png')
plt.show()