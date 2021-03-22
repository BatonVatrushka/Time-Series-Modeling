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

# use the api to pull in the data from FRED
# api_key=329961d88c12ef9d91df6cc8e64c6900
fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

# get the data for E-Commerce Retail Sales
e_commerce = fred.get_series('ECOMNSA')

# get the data for electronic retail sales
electronic_shopping = fred.get_series('MRTSSM4541USN')
electronic_shopping = np.log(electronic_shopping)

# get the data for retail sales: book stores
book_stores = fred.get_series('MRTSSM451211USN')
book_stores = np.log(book_stores)
#----------------------------------------------------
# DECOMPOSE
#----------------------------------------------------
# To decompose data we need to know how often the cycle repeats
# Use the ACF to identify frequency (or use an educated guess)
# the data need to be de-trended by subtracting the rolling average
#----------------------------------------------------
# ELECTRONIC SHOPPING
#----------------------------------------------------
# detrend the data
elec_detrend = electronic_shopping - electronic_shopping.rolling(15).mean()
elec_detrend = elec_detrend.dropna()
# plot the detrended data
elec_detrend.plot()
plt.title('Electronic Shopping Data De-trended')
plt.show()
# plot the acf
plot_acf(elec_detrend, lags=25, zero=False)
plt.title('Electronic Shopping ACF')
plt.show()
# decompose
elec_decomp = seasonal_decompose(electronic_shopping, freq=12)
elec_decomp.plot()
plt.title('Electronic Shopping Seasonal Decomposition')
plt.tight_layout()
plt.show()
#----------------------------------------------------
# BOOK STORES
#----------------------------------------------------
# detrend the data
book_detrend = book_stores - book_stores.rolling(15).mean()
book_detrend = book_detrend.dropna()
# plt the detrend data
book_detrend.plot()
plt.title('Book Store Data De-trended')
plt.show()
# plot the acf
plot_acf(book_detrend, lags=25, zero=False)
plt.title('Book Store ACF')
plt.show()
# decompose
book_decomp = seasonal_decompose(book_stores, freq=12)
book_decomp.plot()
plt.title('Book Stores Seasonal Decomposition')
plt.tight_layout()
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
# ELECTRONIC SHOPPING - SEASONAL DIFFERENCING
#----------------------------------------------------
# S = 12
# create a new variable w/ seasonal differencing
elec_seasonal_diff = electronic_shopping.diff().diff(12).dropna()

# plot it
elec_seasonal_diff.plot(grid=True)
plt.show()

# plot the acf and pacf for non-seasonality
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,6))
# plot the acf
plot_acf(elec_seasonal_diff, lags=11, zero=False, ax=ax1)
# plot the pacf
plot_pacf(elec_seasonal_diff, lags=11, zero=False, ax=ax2)
plt.show()

# plot the acf and pacf for seasonality
lags=[12, 24, 36, 48, 60]
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,6))
# plot the acf
plot_acf(elec_seasonal_diff, lags=lags, zero=False, ax=ax1)
# plot the pacf
plot_pacf(elec_seasonal_diff, lags=lags, zero=False, ax=ax2)
plt.tight_layout()
plt.show()

#----------------------------------------------------
# ELECTRONIC SHOPPING - SEASONAL MODELING
#----------------------------------------------------
results = pm.auto_arima(electronic_shopping,             # data
                        d=1,            # non-seasonal difference order
                        start_p=0,      # initial guess for p
                        start_1=0,      # initial guess for q
                        max_p=3,        # max value of p to test
                        max_q=3,        # max value of q to test
                        seasonal=True,  # is the time series seasonal
                        m=12,            # the seasonal period
                        D=1,            # seasonal difference order
                        start_P=0,      # initial guess for P
                        start_Q=0,      # initial guess for Q
                        max_P=3,        # max value of P to test
                        max_Q=3,        # max value of Q to test
                        information_criterion= 'aic',    # used to select best model
                        trace=True,                     # print results whilst training
                        error_action='ignore',          # ignore orders that don't work
                        stepwise=True,                  # apply intelligent order search
                        )
# print the results
print(results.summary())

# plot the diagnostics
results.plot_diagnostics()
plt.tight_layout()
plt.show()

# ========
# FORECAST
# ========
# create forecast object
model = SARIMAX(electronic_shopping
                , order = (1,1,1)
                , seasonal_order=(3,1,2,12)
                , trend='c')

result = model.fit()

# see the diagnostics
result.plot_diagnostics()
plt.tight_layout()
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
plt.plot(electronic_shopping.index, electronic_shopping, label='past')
# plot the prediction means as line
plt.plot(dates, mean, label='predicted')
# shade between the confidence intervals
plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.7, color='pink')
# plot legend and show fig
plt.legend()
plt.title('Predicted Sales for Electronic Shopping - Logarithmic')
plt.tight_layout()
plt.show()

# PLOT w/ Exponentiation
plt.figure()
# plot past electronic sales
plt.plot(electronic_shopping.index, np.exp(electronic_shopping), label='past')
# plot the prediction means as line
plt.plot(dates, np.exp(mean), label='predicted')
# shade between the confidence intervals
plt.fill_between(dates, np.exp(conf_int.iloc[:,0])
                 , np.exp(conf_int.iloc[:,1]), alpha=0.7, color='pink')
# plot legend and show fig
plt.legend(loc='upper left')
plt.title('Predicted Sales for Electronic Shopping')
plt.tight_layout()
plt.show()

