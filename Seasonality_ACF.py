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

# use the api to pull in the data from FRED
# api_key=329961d88c12ef9d91df6cc8e64c6900
fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

# get the data for E-Commerce Retail Sales
e_commerce = fred.get_series('ECOMNSA')

# get the data for electronic retail sales
electronic_shopping = fred.get_series('MRTSSM4541USN')

# get the data for retail sales: book stores
book_stores = fred.get_series('MRTSSM451211USN')
book_stores.shape
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