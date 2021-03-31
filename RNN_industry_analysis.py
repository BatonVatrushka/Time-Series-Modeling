import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from fredapi import Fred
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from math import sqrt
from statsmodels.tsa.stattools import coint

fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

###########################
# Electronic shopping / Univariate exploration
###########################

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def plot_acf_pacf(series):
    plot_acf(series, ax=plt.gca())
    plt.show()

    plot_pacf(series, ax=plt.gca())
    plt.show()

def test_adf(test_series):
    stationary_result = adfuller(test_series)
    print('ADF Statistic: %f' % stationary_result[0])
    print('p-value: %f' % stationary_result[1])
    print('Critical Values:')
    for key, value in stationary_result[4].items():
    	print('\t%s: %.3f' % (key, value))


electronic_shopping = fred.get_series('MRTSSM4541USN')
electronic_df = electronic_shopping.to_frame(name='electronic')
electronic_df.plot()

# Make stationary
difference(electronic_shopping,1).plot()
np.log10(electronic_shopping).plot()
difference(np.log10(electronic_shopping),1).plot()
difference(np.log10(electronic_shopping),3).plot()
difference(difference(np.log10(electronic_shopping))).plot()

transformed_electronic = pd.Series(difference(difference(np.log10(electronic_shopping))), name='t_electronic')

test_adf(difference(np.log10(electronic_shopping),3))
test_adf(difference(difference(np.log10(electronic_shopping))))

plot_acf_pacf(electronic_shopping)
plot_acf_pacf(difference(np.log10(electronic_shopping),3))


###########################
# Cross Industry Analysis
###########################

book_stores = fred.get_series('MRTSSM451211USN')
book_stores.plot()
book_stores_df = book_stores.to_frame(name = 'books')

all_industries_df = pd.merge(electronic_df, book_stores_df, left_index=True, right_index=True)

building_gardening_sales = fred.get_series("MRTSSM444USN")
building_gardening_sales.plot()

clothing_stores_sales = fred.get_series("MRTSSM448USN")
clothing_stores_sales.plot()

health_personal_care_stores = fred.get_series("MRTSSM446USN")
health_personal_care_stores.plot()
difference(np.log10(health_personal_care_stores)).plot()
difference(difference(np.log10(health_personal_care_stores))).plot()
test_adf(difference(difference(np.log10(health_personal_care_stores))))
transformed_health = pd.Series(difference(difference(np.log10(health_personal_care_stores))), name='t_health')

transformed_df = pd.merge(transformed_electronic, transformed_health, left_index = True, right_index = True)


# Granger Causality Test

from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12

test = 'ssr_chi2test'

def grangers_causality_matrix(data, variables, test = 'ssr_chi2test', verbose=False):

    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]

    dataset.index = [var + '_y' for var in variables]

    return dataset

# Pearson Correlation
def crosscorr(X1, X2, lag=0):
    return X1.corr(X2.shift(lag))

# Spearman Rank Correlation
def spearman_crosscorr(X1, X2, lag=0):
    #measure of a monotonic relationship
    return X1.corr(X2.shift(lag),method="spearman")

# Health vs Electronic shopping

grangers_causality_matrix(transformed_df, variables = transformed_df.columns) 
res = grangercausalitytests(transformed_df, maxlag=15)
transformed_df[['t_electronic','t_health']] = StandardScaler().fit_transform(transformed_df[['t_electronic','t_health']])
coint(transformed_df['t_electronic'],transformed_df['t_health'])


# Beer vs Electronic shopping

beer_sales = fred.get_series("MRTSSM4453USN")
coint(electronic_shopping, beer_sales)
plt.plot(beer_sales)
plt.plot(electronic_shopping)
plt.show()

plt.scatter(electronic_shopping, beer_sales, marker='.')

[crosscorr(electronic_shopping, beer_sales, lag=i) for i in range(24)]
[spearman_crosscorr(electronic_shopping.rank(), beer_sales.rank(), lag=i) for i in range(24)]


# Book stores vs Electronic Shopping

grangers_causality_matrix(all_industries_df, variables = all_industries_df.columns) 
# index 191 is when kindle released
plt.scatter(electronic_shopping[191:], book_stores[191:], marker='.')

[crosscorr(electronic_shopping[191:], book_stores[191:], lag=i) for i in range(24)]
[spearman_crosscorr(electronic_shopping[191:].rank(), book_stores[191:].rank(), lag=i) for i in range(24)]

pd.concat([book_stores, electronic_shopping], axis=1)
grangercausalitytests(pd.concat([book_stores, electronic_shopping], axis=1), 12)





###########################
# Modeling Part II (RNN)
###########################


###################
# LSTM RNN
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = electronic_df
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.97)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5,input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
start = pd.to_datetime('1992-01-01')
rng = pd.date_range(start, periods=350, freq='M')
rngformatted = [i.strftime('%Y%m') for i in rng]
plt.xticks([i for i in range(1, 400,24)],rngformatted[0::24])
plt.xticks(rotation = 90)
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()






###########################
# Model Comparison
###########################

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


electronic_shopping_log = np.log(electronic_shopping)

train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = electronic_shopping_log[0:train_size], electronic_shopping_log[train_size:len(dataset)]
#train, test = electronic_shopping_log[electronic_shopping_log.index < '2020-05-31'], electronic_shopping_log[electronic_shopping_log.index  > '2020-05-31']
test_inv = np.exp(test)

####################
# SARIMA

model = SARIMAX(train
                , order = (1,1,1)
                , seasonal_order=(3,1,2,12)
                , trend='c')

result = model.fit()

# see the diagnostics
result.plot_diagnostics()
plt.tight_layout()
plt.show()

forecast_model1 = result.forecast(len(test))
forecast_model1 = np.exp(forecast_model1)

mse1 = mean_squared_error(test_inv, forecast_model1)
rmse1 = sqrt(mse1)
print('RMSE: %.3f' % rmse1)



plt.plot(pd.Series(electronic_shopping.values, index = [i for i in range(349)]), label = 'Actual')
plt.plot(pd.Series(forecast_model1.values, index = [i for i in range(296, 349)]), label = 'Prediction')
#plt.plot(pd.Series(10**train[150:], index = [i for i in range(150, 343)]), label = 'train')
start = pd.to_datetime('1992-01-01')
rng = pd.date_range(start, periods=350, freq='M')
rngformatted = [i.strftime('%Y%m') for i in rng]
plt.xticks([i for i in range(0, 350,24)],rngformatted[0::24])
plt.xticks(rotation = 90)
plt.legend()
plt.show()



####################
# RNN LStM

numpy.random.seed(7)
dataframe = electronic_df
dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
#train, test = dataset[dataframe.index< '2019-12-31'], dataset[dataframe.index > '2019-12-31']

train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(LSTM(30,input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


start = pd.to_datetime('1992-01-01')
rng = pd.date_range(start, periods=350, freq='M')
rngformatted = [i.strftime('%Y%m') for i in rng]
plt.xticks([i for i in range(1, 400,24)],rngformatted[0::24])
plt.xticks(rotation = 90)
plt.plot(scaler.inverse_transform(dataset), label='Actual')
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, 'r', label='Prediction')
plt.legend()
plt.show()



#plot predictions and real test vals on same line graph
pred_df = pd.DataFrame({'Test': test_inv, 'Prediction_rnn': testPredict.flatten(), 'Prediction_sarima': forecast_model1}, index = electronic_df[-1*(test_size-13):].index)
pred_df.plot()
plt.legend()
plt.ylabel("Sales")
plt.show()