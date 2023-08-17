from flask import Flask
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import requests
import time

app = Flask(__name__)

# Obtaining data from the NBP API 5 years back
start_date = date.today() - timedelta(days=(367 * 5))
end_date = date.today()
delta = timedelta(days=367)
daterange = []
currencies = ['eur', 'usd', 'chf', 'gbp', 'aud', 'cad', 'czk', 'jpy', 'nok', 'dkk']

while start_date < end_date:
    daterange.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta
daterange.append(end_date.strftime("%Y-%m-%d"))

data_list_EUR = []
data_list_USD = []
data_list_CHF = []
data_list_GBP = []
data_list_AUD = []
data_list_CAD = []
data_list_CZK = []
data_list_JPY = []
data_list_NOK = []
data_list_DKK = []
for x in range(len(daterange)):
    for y in range(len(currencies)):
        if x <= (len(daterange) - 2):
            try:
                print('Connecting... ' + str(x) + ' time')
                response_API = requests.get(
                    "https://api.nbp.pl/api/exchangerates/rates/a/"
                    + currencies[y] + "/" + daterange[x] + "/" + daterange[x + 1] + "?format=json")
                if currencies[y] == 'eur':
                    data = response_API.json()['rates']
                    data_list_EUR.append(data)
                elif currencies[y] == 'usd':
                    data = response_API.json()['rates']
                    data_list_USD.append(data)
                elif currencies[y] == 'chf':
                    data = response_API.json()['rates']
                    data_list_CHF.append(data)
                elif currencies[y] == 'gbp':
                    data = response_API.json()['rates']
                    data_list_GBP.append(data)
                elif currencies[y] == 'aud':
                    data = response_API.json()['rates']
                    data_list_AUD.append(data)
                elif currencies[y] == 'cad':
                    data = response_API.json()['rates']
                    data_list_CAD.append(data)
                elif currencies[y] == 'czk':
                    data = response_API.json()['rates']
                    data_list_CZK.append(data)
                elif currencies[y] == 'jpy':
                    data = response_API.json()['rates']
                    data_list_JPY.append(data)
                elif currencies[y] == 'nok':
                    data = response_API.json()['rates']
                    data_list_NOK.append(data)
                elif currencies[y] == 'dkk':
                    data = response_API.json()['rates']
                    data_list_DKK.append(data)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
                print('Reconnecting...')
                time.sleep(5)
                response_API = requests.get(
                    "https://api.nbp.pl/api/exchangerates/rates/a/"
                    + currencies[y] + "/" + daterange[x] + "/" + daterange[x + 1] + "?format=json")
                if currencies[y] == 'eur':
                    data = response_API.json()['rates']
                    data_list_EUR.append(data)
                elif currencies[y] == 'usd':
                    data = response_API.json()['rates']
                    data_list_USD.append(data)
                elif currencies[y] == 'chf':
                    data = response_API.json()['rates']
                    data_list_CHF.append(data)
                elif currencies[y] == 'gbp':
                    data = response_API.json()['rates']
                    data_list_GBP.append(data)
                elif currencies[y] == 'aud':
                    data = response_API.json()['rates']
                    data_list_AUD.append(data)
                elif currencies[y] == 'cad':
                    data = response_API.json()['rates']
                    data_list_CAD.append(data)
                elif currencies[y] == 'czk':
                    data = response_API.json()['rates']
                    data_list_CZK.append(data)
                elif currencies[y] == 'jpy':
                    data = response_API.json()['rates']
                    data_list_JPY.append(data)
                elif currencies[y] == 'nok':
                    data = response_API.json()['rates']
                    data_list_NOK.append(data)
                elif currencies[y] == 'dkk':
                    data = response_API.json()['rates']
                    data_list_DKK.append(data)

merged_data_list_EUR = list(itertools.chain(*data_list_EUR))
merged_data_list_USD = list(itertools.chain(*data_list_USD))
merged_data_list_CHF = list(itertools.chain(*data_list_CHF))
merged_data_list_GBP = list(itertools.chain(*data_list_GBP))
merged_data_list_AUD = list(itertools.chain(*data_list_AUD))
merged_data_list_CAD = list(itertools.chain(*data_list_CAD))
merged_data_list_CZK = list(itertools.chain(*data_list_CZK))
merged_data_list_JPY = list(itertools.chain(*data_list_JPY))
merged_data_list_NOK = list(itertools.chain(*data_list_NOK))
merged_data_list_DKK = list(itertools.chain(*data_list_DKK))

df_EUR = pd.DataFrame(merged_data_list_EUR)
df_USD = pd.DataFrame(merged_data_list_USD)
df_CHF = pd.DataFrame(merged_data_list_CHF)
df_GBP = pd.DataFrame(merged_data_list_GBP)
df_AUD = pd.DataFrame(merged_data_list_AUD)
df_CAD = pd.DataFrame(merged_data_list_CAD)
df_CZK = pd.DataFrame(merged_data_list_CZK)
df_JPY = pd.DataFrame(merged_data_list_JPY)
df_NOK = pd.DataFrame(merged_data_list_NOK)
df_DKK = pd.DataFrame(merged_data_list_DKK)

df_EUR_1 = df_EUR.reset_index()['mid']
df_USD_2 = df_USD.reset_index()['mid']
df_CHF_3 = df_CHF.reset_index()['mid']
df_GBP_4 = df_GBP.reset_index()['mid']
df_AUD_5 = df_AUD.reset_index()['mid']
df_CAD_6 = df_CAD.reset_index()['mid']
df_CZK_7 = df_CZK.reset_index()['mid']
df_JPY_8 = df_JPY.reset_index()['mid']
df_NOK_9 = df_NOK.reset_index()['mid']
df_DKK_10 = df_DKK.reset_index()['mid']

# LSTM are sensitive to the scale of the data. so we apply MinMax scaler
scaler_EUR = MinMaxScaler(feature_range=(0, 1))
scaler_USD = MinMaxScaler(feature_range=(0, 1))
scaler_CHF = MinMaxScaler(feature_range=(0, 1))
scaler_GBP = MinMaxScaler(feature_range=(0, 1))
scaler_AUD = MinMaxScaler(feature_range=(0, 1))
scaler_CAD = MinMaxScaler(feature_range=(0, 1))
scaler_CZK = MinMaxScaler(feature_range=(0, 1))
scaler_JPY = MinMaxScaler(feature_range=(0, 1))
scaler_NOK = MinMaxScaler(feature_range=(0, 1))
scaler_DKK = MinMaxScaler(feature_range=(0, 1))

df_EUR_1 = scaler_EUR.fit_transform(np.array(df_EUR_1).reshape(-1, 1))
df_USD_2 = scaler_USD.fit_transform(np.array(df_USD_2).reshape(-1, 1))
df_CHF_3 = scaler_CHF.fit_transform(np.array(df_CHF_3).reshape(-1, 1))
df_GBP_4 = scaler_GBP.fit_transform(np.array(df_GBP_4).reshape(-1, 1))
df_AUD_5 = scaler_AUD.fit_transform(np.array(df_AUD_5).reshape(-1, 1))
df_CAD_6 = scaler_CAD.fit_transform(np.array(df_CAD_6).reshape(-1, 1))
df_CZK_7 = scaler_CZK.fit_transform(np.array(df_CZK_7).reshape(-1, 1))
df_JPY_8 = scaler_JPY.fit_transform(np.array(df_JPY_8).reshape(-1, 1))
df_NOK_9 = scaler_NOK.fit_transform(np.array(df_NOK_9).reshape(-1, 1))
df_DKK_10 = scaler_DKK.fit_transform(np.array(df_DKK_10).reshape(-1, 1))

# Splitting dataset into train and test split
training_size_EUR = int(len(df_EUR_1) * 0.65)
training_size_USD = int(len(df_USD_2) * 0.65)
training_size_CHF = int(len(df_CHF_3) * 0.65)
training_size_GBP = int(len(df_GBP_4) * 0.65)
training_size_AUD = int(len(df_AUD_5) * 0.65)
training_size_CAD = int(len(df_CAD_6) * 0.65)
training_size_CZK = int(len(df_CZK_7) * 0.65)
training_size_JPY = int(len(df_JPY_8) * 0.65)
training_size_NOK = int(len(df_NOK_9) * 0.65)
training_size_DKK = int(len(df_DKK_10) * 0.65)

test_size_EUR = len(df_EUR_1) - training_size_EUR
test_size_USD = len(df_USD_2) - training_size_USD
test_size_CHF = len(df_CHF_3) - training_size_CHF
test_size_GBP = len(df_GBP_4) - training_size_GBP
test_size_AUD = len(df_AUD_5) - training_size_AUD
test_size_CAD = len(df_CAD_6) - training_size_CAD
test_size_CZK = len(df_CZK_7) - training_size_CZK
test_size_JPY = len(df_JPY_8) - training_size_JPY
test_size_NOK = len(df_NOK_9) - training_size_NOK
test_size_DKK = len(df_DKK_10) - training_size_DKK

train_data_EUR, test_data_EUR = df_EUR_1[0:training_size_EUR, :], df_EUR_1[training_size_EUR:len(df_EUR_1), :1]
train_data_USD, test_data_USD = df_USD_2[0:training_size_USD, :], df_USD_2[training_size_USD:len(df_USD_2), :1]
train_data_CHF, test_data_CHF = df_CHF_3[0:training_size_CHF, :], df_CHF_3[training_size_CHF:len(df_CHF_3), :1]
train_data_GBP, test_data_GBP = df_GBP_4[0:training_size_GBP, :], df_GBP_4[training_size_GBP:len(df_GBP_4), :1]
train_data_AUD, test_data_AUD = df_AUD_5[0:training_size_AUD, :], df_AUD_5[training_size_AUD:len(df_AUD_5), :1]
train_data_CAD, test_data_CAD = df_CAD_6[0:training_size_CAD, :], df_CAD_6[training_size_CAD:len(df_CAD_6), :1]
train_data_CZK, test_data_CZK = df_CZK_7[0:training_size_CZK, :], df_CZK_7[training_size_CZK:len(df_CZK_7), :1]
train_data_JPY, test_data_JPY = df_JPY_8[0:training_size_JPY, :], df_JPY_8[training_size_JPY:len(df_JPY_8), :1]
train_data_NOK, test_data_NOK = df_NOK_9[0:training_size_NOK, :], df_NOK_9[training_size_NOK:len(df_NOK_9), :1]
train_data_DKK, test_data_DKK = df_DKK_10[0:training_size_DKK, :], df_DKK_10[training_size_DKK:len(df_DKK_10), :1]

# RESHAPING TRAIN AND TEST DATA
time_step_EUR = 100
time_step_USD = 100
time_step_CHF = 100
time_step_GBP = 100
time_step_AUD = 100
time_step_CAD = 100
time_step_CZK = 100
time_step_JPY = 100
time_step_NOK = 100
time_step_DKK = 100

X_train_EUR, y_train_EUR = preprocessing.new_dataset(train_data_EUR, time_step_EUR)
X_train_USD, y_train_USD = preprocessing.new_dataset(train_data_USD, time_step_USD)
X_train_CHF, y_train_CHF = preprocessing.new_dataset(train_data_CHF, time_step_CHF)
X_train_GBP, y_train_GBP = preprocessing.new_dataset(train_data_GBP, time_step_GBP)
X_train_AUD, y_train_AUD = preprocessing.new_dataset(train_data_AUD, time_step_AUD)
X_train_CAD, y_train_CAD = preprocessing.new_dataset(train_data_CAD, time_step_CAD)
X_train_CZK, y_train_CZK = preprocessing.new_dataset(train_data_CZK, time_step_CZK)
X_train_JPY, y_train_JPY = preprocessing.new_dataset(train_data_JPY, time_step_JPY)
X_train_NOK, y_train_NOK = preprocessing.new_dataset(train_data_NOK, time_step_NOK)
X_train_DKK, y_train_DKK = preprocessing.new_dataset(train_data_DKK, time_step_DKK)

X_test_EUR, ytest_EUR = preprocessing.new_dataset(test_data_EUR, time_step_EUR)
X_test_USD, ytest_USD = preprocessing.new_dataset(test_data_USD, time_step_USD)
X_test_CHF, ytest_CHF = preprocessing.new_dataset(test_data_CHF, time_step_CHF)
X_test_GBP, ytest_GBP = preprocessing.new_dataset(test_data_GBP, time_step_GBP)
X_test_AUD, ytest_AUD = preprocessing.new_dataset(test_data_AUD, time_step_AUD)
X_test_CAD, ytest_CAD = preprocessing.new_dataset(test_data_CAD, time_step_CAD)
X_test_CZK, ytest_CZK = preprocessing.new_dataset(test_data_CZK, time_step_CZK)
X_test_JPY, ytest_JPY = preprocessing.new_dataset(test_data_JPY, time_step_JPY)
X_test_NOK, ytest_NOK = preprocessing.new_dataset(test_data_NOK, time_step_NOK)
X_test_DKK, ytest_DKK = preprocessing.new_dataset(test_data_DKK, time_step_DKK)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train_EUR = X_train_EUR.reshape(X_train_EUR.shape[0], X_train_EUR.shape[1], 1)
X_train_USD = X_train_USD.reshape(X_train_USD.shape[0], X_train_USD.shape[1], 1)
X_train_CHF = X_train_CHF.reshape(X_train_CHF.shape[0], X_train_CHF.shape[1], 1)
X_train_GBP = X_train_GBP.reshape(X_train_GBP.shape[0], X_train_GBP.shape[1], 1)
X_train_AUD = X_train_AUD.reshape(X_train_AUD.shape[0], X_train_AUD.shape[1], 1)
X_train_CAD = X_train_CAD.reshape(X_train_CAD.shape[0], X_train_CAD.shape[1], 1)
X_train_CZK = X_train_CZK.reshape(X_train_CZK.shape[0], X_train_CZK.shape[1], 1)
X_train_JPY = X_train_JPY.reshape(X_train_JPY.shape[0], X_train_JPY.shape[1], 1)
X_train_NOK = X_train_NOK.reshape(X_train_NOK.shape[0], X_train_NOK.shape[1], 1)
X_train_DKK = X_train_DKK.reshape(X_train_DKK.shape[0], X_train_DKK.shape[1], 1)

X_test_EUR = X_test_EUR.reshape(X_test_EUR.shape[0], X_test_EUR.shape[1], 1)
X_test_USD = X_test_USD.reshape(X_test_USD.shape[0], X_test_USD.shape[1], 1)
X_test_CHF = X_test_CHF.reshape(X_test_CHF.shape[0], X_test_CHF.shape[1], 1)
X_test_GBP = X_test_GBP.reshape(X_test_GBP.shape[0], X_test_GBP.shape[1], 1)
X_test_AUD = X_test_AUD.reshape(X_test_AUD.shape[0], X_test_AUD.shape[1], 1)
X_test_CAD = X_test_CAD.reshape(X_test_CAD.shape[0], X_test_CAD.shape[1], 1)
X_test_CZK = X_test_CZK.reshape(X_test_CZK.shape[0], X_test_CZK.shape[1], 1)
X_test_JPY = X_test_JPY.reshape(X_test_JPY.shape[0], X_test_JPY.shape[1], 1)
X_test_NOK = X_test_NOK.reshape(X_test_NOK.shape[0], X_test_NOK.shape[1], 1)
X_test_DKK = X_test_DKK.reshape(X_test_DKK.shape[0], X_test_DKK.shape[1], 1)

# Create the Stacked LSTM model
model_EUR = Sequential()
model_USD = Sequential()
model_CHF = Sequential()
model_GBP = Sequential()
model_AUD = Sequential()
model_CAD = Sequential()
model_CZK = Sequential()
model_JPY = Sequential()
model_NOK = Sequential()
model_DKK = Sequential()

model_EUR.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_USD.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_CHF.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_GBP.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_AUD.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_CAD.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_CZK.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_JPY.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_NOK.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model_DKK.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))

model_EUR.add(LSTM(50, return_sequences=True))
model_USD.add(LSTM(50, return_sequences=True))
model_CHF.add(LSTM(50, return_sequences=True))
model_GBP.add(LSTM(50, return_sequences=True))
model_AUD.add(LSTM(50, return_sequences=True))
model_CAD.add(LSTM(50, return_sequences=True))
model_CZK.add(LSTM(50, return_sequences=True))
model_JPY.add(LSTM(50, return_sequences=True))
model_NOK.add(LSTM(50, return_sequences=True))
model_DKK.add(LSTM(50, return_sequences=True))

model_EUR.add(LSTM(50))
model_USD.add(LSTM(50))
model_CHF.add(LSTM(50))
model_GBP.add(LSTM(50))
model_AUD.add(LSTM(50))
model_CAD.add(LSTM(50))
model_CZK.add(LSTM(50))
model_JPY.add(LSTM(50))
model_NOK.add(LSTM(50))
model_DKK.add(LSTM(50))

model_EUR.add(Dense(1))
model_USD.add(Dense(1))
model_CHF.add(Dense(1))
model_GBP.add(Dense(1))
model_AUD.add(Dense(1))
model_CAD.add(Dense(1))
model_CZK.add(Dense(1))
model_JPY.add(Dense(1))
model_NOK.add(Dense(1))
model_DKK.add(Dense(1))

model_EUR.compile(loss='mean_squared_error', optimizer='adam')
model_USD.compile(loss='mean_squared_error', optimizer='adam')
model_CHF.compile(loss='mean_squared_error', optimizer='adam')
model_GBP.compile(loss='mean_squared_error', optimizer='adam')
model_AUD.compile(loss='mean_squared_error', optimizer='adam')
model_CAD.compile(loss='mean_squared_error', optimizer='adam')
model_CZK.compile(loss='mean_squared_error', optimizer='adam')
model_JPY.compile(loss='mean_squared_error', optimizer='adam')
model_NOK.compile(loss='mean_squared_error', optimizer='adam')
model_DKK.compile(loss='mean_squared_error', optimizer='adam')

model_EUR.summary()
model_USD.summary()
model_CHF.summary()
model_GBP.summary()
model_AUD.summary()
model_CAD.summary()
model_CZK.summary()
model_JPY.summary()
model_NOK.summary()
model_DKK.summary()

model_EUR.fit(X_train_EUR, y_train_EUR, validation_data=(X_test_EUR, ytest_EUR), epochs=10, batch_size=64, verbose=1)
model_USD.fit(X_train_USD, y_train_USD, validation_data=(X_test_USD, ytest_USD), epochs=10, batch_size=64, verbose=1)
model_CHF.fit(X_train_CHF, y_train_CHF, validation_data=(X_test_CHF, ytest_CHF), epochs=10, batch_size=64, verbose=1)
model_GBP.fit(X_train_GBP, y_train_GBP, validation_data=(X_test_GBP, ytest_GBP), epochs=10, batch_size=64, verbose=1)
model_AUD.fit(X_train_AUD, y_train_AUD, validation_data=(X_test_AUD, ytest_AUD), epochs=10, batch_size=64, verbose=1)
model_CAD.fit(X_train_CAD, y_train_CAD, validation_data=(X_test_CAD, ytest_CAD), epochs=10, batch_size=64, verbose=1)
model_CZK.fit(X_train_CZK, y_train_CZK, validation_data=(X_test_CZK, ytest_CZK), epochs=10, batch_size=64, verbose=1)
model_JPY.fit(X_train_JPY, y_train_JPY, validation_data=(X_test_JPY, ytest_JPY), epochs=10, batch_size=64, verbose=1)
model_NOK.fit(X_train_NOK, y_train_NOK, validation_data=(X_test_NOK, ytest_NOK), epochs=10, batch_size=64, verbose=1)
model_DKK.fit(X_train_DKK, y_train_DKK, validation_data=(X_test_DKK, ytest_DKK), epochs=10, batch_size=64, verbose=1)

# Lets Do the prediction and check performance metrics
train_predict_EUR = model_EUR.predict(X_train_EUR)
train_predict_USD = model_USD.predict(X_train_USD)
train_predict_CHF = model_CHF.predict(X_train_CHF)
train_predict_GBP = model_GBP.predict(X_train_GBP)
train_predict_AUD = model_AUD.predict(X_train_AUD)
train_predict_CAD = model_CAD.predict(X_train_CAD)
train_predict_CZK = model_CZK.predict(X_train_CZK)
train_predict_JPY = model_JPY.predict(X_train_JPY)
train_predict_NOK = model_NOK.predict(X_train_NOK)
train_predict_DKK = model_DKK.predict(X_train_DKK)

test_predict_EUR = model_EUR.predict(X_test_EUR)
test_predict_USD = model_USD.predict(X_test_USD)
test_predict_CHF = model_CHF.predict(X_test_CHF)
test_predict_GBP = model_GBP.predict(X_test_GBP)
test_predict_AUD = model_AUD.predict(X_test_AUD)
test_predict_CAD = model_CAD.predict(X_test_CAD)
test_predict_CZK = model_CZK.predict(X_test_CZK)
test_predict_JPY = model_JPY.predict(X_test_JPY)
test_predict_NOK = model_NOK.predict(X_test_NOK)
test_predict_DKK = model_DKK.predict(X_test_DKK)

# Transformback to original form
train_predict_EUR = scaler_EUR.inverse_transform(train_predict_EUR)
train_predict_USD = scaler_USD.inverse_transform(train_predict_USD)
train_predict_CHF = scaler_CHF.inverse_transform(train_predict_CHF)
train_predict_GBP = scaler_GBP.inverse_transform(train_predict_GBP)
train_predict_AUD = scaler_AUD.inverse_transform(train_predict_AUD)
train_predict_CAD = scaler_CAD.inverse_transform(train_predict_CAD)
train_predict_CZK = scaler_CZK.inverse_transform(train_predict_CZK)
train_predict_JPY = scaler_JPY.inverse_transform(train_predict_JPY)
train_predict_NOK = scaler_NOK.inverse_transform(train_predict_NOK)
train_predict_DKK = scaler_DKK.inverse_transform(train_predict_DKK)

test_predict_EUR = scaler_EUR.inverse_transform(test_predict_EUR)
test_predict_USD = scaler_USD.inverse_transform(test_predict_USD)
test_predict_CHF = scaler_CHF.inverse_transform(test_predict_CHF)
test_predict_GBP = scaler_GBP.inverse_transform(test_predict_GBP)
test_predict_AUD = scaler_AUD.inverse_transform(test_predict_AUD)
test_predict_CAD = scaler_CAD.inverse_transform(test_predict_CAD)
test_predict_CZK = scaler_CZK.inverse_transform(test_predict_CZK)
test_predict_JPY = scaler_JPY.inverse_transform(test_predict_JPY)
test_predict_NOK = scaler_NOK.inverse_transform(test_predict_NOK)
test_predict_DKK = scaler_DKK.inverse_transform(test_predict_DKK)

# Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train_EUR, train_predict_EUR))
math.sqrt(mean_squared_error(y_train_USD, train_predict_USD))
math.sqrt(mean_squared_error(y_train_CHF, train_predict_CHF))
math.sqrt(mean_squared_error(y_train_GBP, train_predict_GBP))
math.sqrt(mean_squared_error(y_train_AUD, train_predict_AUD))
math.sqrt(mean_squared_error(y_train_CAD, train_predict_CAD))
math.sqrt(mean_squared_error(y_train_CZK, train_predict_CZK))
math.sqrt(mean_squared_error(y_train_JPY, train_predict_JPY))
math.sqrt(mean_squared_error(y_train_NOK, train_predict_NOK))
math.sqrt(mean_squared_error(y_train_DKK, train_predict_DKK))

# Test Data RMSE
math.sqrt(mean_squared_error(ytest_EUR, test_predict_EUR))
math.sqrt(mean_squared_error(ytest_USD, test_predict_USD))
math.sqrt(mean_squared_error(ytest_CHF, test_predict_CHF))
math.sqrt(mean_squared_error(ytest_GBP, test_predict_GBP))
math.sqrt(mean_squared_error(ytest_AUD, test_predict_AUD))
math.sqrt(mean_squared_error(ytest_CAD, test_predict_CAD))
math.sqrt(mean_squared_error(ytest_CZK, test_predict_CZK))
math.sqrt(mean_squared_error(ytest_JPY, test_predict_JPY))
math.sqrt(mean_squared_error(ytest_NOK, test_predict_NOK))
math.sqrt(mean_squared_error(ytest_DKK, test_predict_DKK))

# print(len(test_data_EUR))  # 446

if len(test_data_EUR) == 446:
    x_input_EUR = test_data_EUR[346:].reshape(1, -1)
else:
    x_input_EUR = test_data_EUR[345:].reshape(1, -1)

if len(test_data_USD) == 446:
    x_input_USD = test_data_USD[346:].reshape(1, -1)
else:
    x_input_USD = test_data_USD[345:].reshape(1, -1)

if len(test_data_CHF) == 446:
    x_input_CHF = test_data_CHF[346:].reshape(1, -1)
else:
    x_input_CHF = test_data_CHF[345:].reshape(1, -1)

if len(test_data_GBP) == 446:
    x_input_GBP = test_data_GBP[346:].reshape(1, -1)
else:
    x_input_GBP = test_data_GBP[345:].reshape(1, -1)

if len(test_data_AUD) == 446:
    x_input_AUD = test_data_AUD[346:].reshape(1, -1)
else:
    x_input_AUD = test_data_AUD[345:].reshape(1, -1)

if len(test_data_CAD) == 446:
    x_input_CAD = test_data_CAD[346:].reshape(1, -1)
else:
    x_input_CAD = test_data_CAD[345:].reshape(1, -1)

if len(test_data_CZK) == 446:
    x_input_CZK = test_data_CZK[346:].reshape(1, -1)
else:
    x_input_CZK = test_data_CZK[345:].reshape(1, -1)

if len(test_data_JPY) == 446:
    x_input_JPY = test_data_JPY[346:].reshape(1, -1)
else:
    x_input_JPY = test_data_JPY[345:].reshape(1, -1)

if len(test_data_NOK) == 446:
    x_input_NOK = test_data_NOK[346:].reshape(1, -1)
else:
    x_input_NOK = test_data_NOK[345:].reshape(1, -1)

if len(test_data_DKK) == 446:
    x_input_DKK = test_data_DKK[346:].reshape(1, -1)
else:
    x_input_DKK = test_data_DKK[345:].reshape(1, -1)

temp_input_EUR = list(x_input_EUR)
temp_input_USD = list(x_input_USD)
temp_input_CHF = list(x_input_CHF)
temp_input_GBP = list(x_input_GBP)
temp_input_AUD = list(x_input_AUD)
temp_input_CAD = list(x_input_CAD)
temp_input_CZK = list(x_input_CZK)
temp_input_JPY = list(x_input_JPY)
temp_input_NOK = list(x_input_NOK)
temp_input_DKK = list(x_input_DKK)

temp_input_EUR = temp_input_EUR[0].tolist()
temp_input_USD = temp_input_USD[0].tolist()
temp_input_CHF = temp_input_CHF[0].tolist()
temp_input_GBP = temp_input_GBP[0].tolist()
temp_input_AUD = temp_input_AUD[0].tolist()
temp_input_CAD = temp_input_CAD[0].tolist()
temp_input_CZK = temp_input_CZK[0].tolist()
temp_input_JPY = temp_input_JPY[0].tolist()
temp_input_NOK = temp_input_NOK[0].tolist()
temp_input_DKK = temp_input_DKK[0].tolist()

# Demonstrate prediction for next 30 days
lst_output_EUR = []
n_steps_EUR = 100
i_EUR = 0
while i_EUR < 30:
    if len(temp_input_EUR) > 100:
        # EUR
        x_input_EUR = np.array(temp_input_EUR[1:])
        x_input_EUR = x_input_EUR.reshape(1, -1)
        x_input_EUR = x_input_EUR.reshape((1, n_steps_EUR, 1))
        yhat_EUR = model_EUR.predict(x_input_EUR, verbose=0)
        temp_input_EUR.extend(yhat_EUR[0].tolist())
        temp_input_EUR = temp_input_EUR[1:]
        lst_output_EUR.extend(yhat_EUR.tolist())
        i_EUR = i_EUR + 1
    else:
        # EUR
        x_input_EUR = x_input_EUR.reshape((1, n_steps_EUR, 1))
        yhat_EUR = model_EUR.predict(x_input_EUR, verbose=0)
        temp_input_EUR.extend(yhat_EUR[0].tolist())
        lst_output_EUR.extend(yhat_EUR.tolist())
        i_EUR = i_EUR + 1

lst_output_USD = []
n_steps_USD = 100
i_USD = 0
while i_USD < 30:
    if len(temp_input_USD) > 100:
        # USD
        x_input_USD = np.array(temp_input_USD[1:])
        x_input_USD = x_input_USD.reshape(1, -1)
        x_input_USD = x_input_USD.reshape((1, n_steps_USD, 1))
        yhat_USD = model_USD.predict(x_input_USD, verbose=0)
        temp_input_USD.extend(yhat_USD[0].tolist())
        temp_input_USD = temp_input_USD[1:]
        lst_output_USD.extend(yhat_USD.tolist())
        i_USD = i_USD + 1
    else:
        # USD
        x_input_USD = x_input_USD.reshape((1, n_steps_USD, 1))
        yhat_USD = model_USD.predict(x_input_USD, verbose=0)
        temp_input_USD.extend(yhat_USD[0].tolist())
        lst_output_USD.extend(yhat_USD.tolist())
        i_USD = i_USD + 1

lst_output_CHF = []
n_steps_CHF = 100
i_CHF = 0
while i_CHF < 30:
    if len(temp_input_CHF) > 100:
        # CHF
        x_input_CHF = np.array(temp_input_CHF[1:])
        x_input_CHF = x_input_CHF.reshape(1, -1)
        x_input_CHF = x_input_CHF.reshape((1, n_steps_CHF, 1))
        yhat_CHF = model_CHF.predict(x_input_CHF, verbose=0)
        temp_input_CHF.extend(yhat_CHF[0].tolist())
        temp_input_CHF = temp_input_CHF[1:]
        lst_output_CHF.extend(yhat_CHF.tolist())
        i_CHF = i_CHF + 1
    else:
        # CHF
        x_input_CHF = x_input_CHF.reshape((1, n_steps_CHF, 1))
        yhat_CHF = model_CHF.predict(x_input_CHF, verbose=0)
        temp_input_CHF.extend(yhat_CHF[0].tolist())
        lst_output_CHF.extend(yhat_CHF.tolist())
        i_CHF = i_CHF + 1

lst_output_GBP = []
n_steps_GBP = 100
i_GBP = 0
while i_GBP < 30:
    if len(temp_input_GBP) > 100:
        # GBP
        x_input_GBP = np.array(temp_input_GBP[1:])
        x_input_GBP = x_input_GBP.reshape(1, -1)
        x_input_GBP = x_input_GBP.reshape((1, n_steps_GBP, 1))
        yhat_GBP = model_GBP.predict(x_input_GBP, verbose=0)
        temp_input_GBP.extend(yhat_GBP[0].tolist())
        temp_input_GBP = temp_input_GBP[1:]
        lst_output_GBP.extend(yhat_GBP.tolist())
        i_GBP = i_GBP + 1
    else:
        # GBP
        x_input_GBP = x_input_GBP.reshape((1, n_steps_GBP, 1))
        yhat_GBP = model_GBP.predict(x_input_GBP, verbose=0)
        temp_input_GBP.extend(yhat_GBP[0].tolist())
        lst_output_GBP.extend(yhat_GBP.tolist())
        i_GBP = i_GBP + 1

lst_output_AUD = []
n_steps_AUD = 100
i_AUD = 0
while i_AUD < 30:
    if len(temp_input_AUD) > 100:
        # AUD
        x_input_AUD = np.array(temp_input_AUD[1:])
        x_input_AUD = x_input_AUD.reshape(1, -1)
        x_input_AUD = x_input_AUD.reshape((1, n_steps_AUD, 1))
        yhat_AUD = model_AUD.predict(x_input_AUD, verbose=0)
        temp_input_AUD.extend(yhat_AUD[0].tolist())
        temp_input_AUD = temp_input_AUD[1:]
        lst_output_AUD.extend(yhat_AUD.tolist())
        i_AUD = i_AUD + 1
    else:
        # AUD
        x_input_AUD = x_input_AUD.reshape((1, n_steps_AUD, 1))
        yhat_AUD = model_AUD.predict(x_input_AUD, verbose=0)
        temp_input_AUD.extend(yhat_AUD[0].tolist())
        lst_output_AUD.extend(yhat_AUD.tolist())
        i_AUD = i_AUD + 1

lst_output_CAD = []
n_steps_CAD = 100
i_CAD = 0
while i_CAD < 30:
    if len(temp_input_CAD) > 100:
        # CAD
        x_input_CAD = np.array(temp_input_CAD[1:])
        x_input_CAD = x_input_CAD.reshape(1, -1)
        x_input_CAD = x_input_CAD.reshape((1, n_steps_CAD, 1))
        yhat_CAD = model_CAD.predict(x_input_CAD, verbose=0)
        temp_input_CAD.extend(yhat_CAD[0].tolist())
        temp_input_CAD = temp_input_CAD[1:]
        lst_output_CAD.extend(yhat_CAD.tolist())
        i_CAD = i_CAD + 1
    else:
        # CAD
        x_input_CAD = x_input_CAD.reshape((1, n_steps_CAD, 1))
        yhat_CAD = model_CAD.predict(x_input_CAD, verbose=0)
        temp_input_CAD.extend(yhat_CAD[0].tolist())
        lst_output_CAD.extend(yhat_CAD.tolist())
        i_CAD = i_CAD + 1

lst_output_CZK = []
n_steps_CZK = 100
i_CZK = 0
while i_CZK < 30:
    if len(temp_input_CZK) > 100:
        # CZK
        x_input_CZK = np.array(temp_input_CZK[1:])
        x_input_CZK = x_input_CZK.reshape(1, -1)
        x_input_CZK = x_input_CZK.reshape((1, n_steps_CZK, 1))
        yhat_CZK = model_CZK.predict(x_input_CZK, verbose=0)
        temp_input_CZK.extend(yhat_CZK[0].tolist())
        temp_input_CZK = temp_input_CZK[1:]
        lst_output_CZK.extend(yhat_CZK.tolist())
        i_CZK = i_CZK + 1
    else:
        # CZK
        x_input_CZK = x_input_CZK.reshape((1, n_steps_CZK, 1))
        yhat_CZK = model_CZK.predict(x_input_CZK, verbose=0)
        temp_input_CZK.extend(yhat_CZK[0].tolist())
        lst_output_CZK.extend(yhat_CZK.tolist())
        i_CZK = i_CZK + 1

lst_output_JPY = []
n_steps_JPY = 100
i_JPY = 0
while i_JPY < 30:
    if len(temp_input_JPY) > 100:
        # JPY
        x_input_JPY = np.array(temp_input_JPY[1:])
        x_input_JPY = x_input_JPY.reshape(1, -1)
        x_input_JPY = x_input_JPY.reshape((1, n_steps_JPY, 1))
        yhat_JPY = model_JPY.predict(x_input_JPY, verbose=0)
        temp_input_JPY.extend(yhat_JPY[0].tolist())
        temp_input_JPY = temp_input_JPY[1:]
        lst_output_JPY.extend(yhat_JPY.tolist())
        i_JPY = i_JPY + 1
    else:
        # JPY
        x_input_JPY = x_input_JPY.reshape((1, n_steps_JPY, 1))
        yhat_JPY = model_JPY.predict(x_input_JPY, verbose=0)
        temp_input_JPY.extend(yhat_JPY[0].tolist())
        lst_output_JPY.extend(yhat_JPY.tolist())
        i_JPY = i_JPY + 1

lst_output_NOK = []
n_steps_NOK = 100
i_NOK = 0
while i_NOK < 30:
    if len(temp_input_NOK) > 100:
        # NOK
        x_input_NOK = np.array(temp_input_NOK[1:])
        x_input_NOK = x_input_NOK.reshape(1, -1)
        x_input_NOK = x_input_NOK.reshape((1, n_steps_NOK, 1))
        yhat_NOK = model_NOK.predict(x_input_NOK, verbose=0)
        temp_input_NOK.extend(yhat_NOK[0].tolist())
        temp_input_NOK = temp_input_NOK[1:]
        lst_output_NOK.extend(yhat_NOK.tolist())
        i_NOK = i_NOK + 1
    else:
        # NOK
        x_input_NOK = x_input_NOK.reshape((1, n_steps_NOK, 1))
        yhat_NOK = model_NOK.predict(x_input_NOK, verbose=0)
        temp_input_NOK.extend(yhat_NOK[0].tolist())
        lst_output_NOK.extend(yhat_NOK.tolist())
        i_NOK = i_NOK + 1

lst_output_DKK = []
n_steps_DKK = 100
i_DKK = 0
while i_DKK < 30:
    if len(temp_input_DKK) > 100:
        # DKK
        x_input_DKK = np.array(temp_input_DKK[1:])
        x_input_DKK = x_input_DKK.reshape(1, -1)
        x_input_DKK = x_input_DKK.reshape((1, n_steps_DKK, 1))
        yhat_DKK = model_DKK.predict(x_input_DKK, verbose=0)
        temp_input_DKK.extend(yhat_DKK[0].tolist())
        temp_input_DKK = temp_input_DKK[1:]
        lst_output_DKK.extend(yhat_DKK.tolist())
        i_DKK = i_DKK + 1
    else:
        # DKK
        x_input_DKK = x_input_DKK.reshape((1, n_steps_DKK, 1))
        yhat_DKK = model_DKK.predict(x_input_DKK, verbose=0)
        temp_input_DKK.extend(yhat_DKK[0].tolist())
        lst_output_DKK.extend(yhat_DKK.tolist())
        i_DKK = i_DKK + 1

predictions_EUR = scaler_EUR.inverse_transform(lst_output_EUR).tolist()
predictions_USD = scaler_USD.inverse_transform(lst_output_USD).tolist()
predictions_CHF = scaler_CHF.inverse_transform(lst_output_CHF).tolist()
predictions_GBP = scaler_GBP.inverse_transform(lst_output_GBP).tolist()
predictions_AUD = scaler_AUD.inverse_transform(lst_output_AUD).tolist()
predictions_CAD = scaler_CAD.inverse_transform(lst_output_CAD).tolist()
predictions_CZK = scaler_CZK.inverse_transform(lst_output_CZK).tolist()
predictions_JPY = scaler_JPY.inverse_transform(lst_output_JPY).tolist()
predictions_NOK = scaler_NOK.inverse_transform(lst_output_NOK).tolist()
predictions_DKK = scaler_DKK.inverse_transform(lst_output_DKK).tolist()

@app.route('/EUR', methods=['GET'])
def get_eur_pred():
    return predictions_EUR

@app.route('/USD', methods=['GET'])
def get_usd_pred():
    return predictions_USD

@app.route('/CHF', methods=['GET'])
def get_chf_pred():
    return predictions_CHF

@app.route('/GBP', methods=['GET'])
def get_gbp_pred():
    return predictions_GBP

@app.route('/AUD', methods=['GET'])
def get_aud_pred():
    return predictions_AUD

@app.route('/CAD', methods=['GET'])
def get_cad_pred():
    return predictions_CAD

@app.route('/CZK', methods=['GET'])
def get_czk_pred():
    return predictions_CZK

@app.route('/JPY', methods=['GET'])
def get_jpy_pred():
    return predictions_JPY

@app.route('/NOK', methods=['GET'])
def get_nok_pred():
    return predictions_NOK

@app.route('/DKK', methods=['GET'])
def get_dkk_pred():
    return predictions_DKK

if __name__ == "__main__":
    # app.run(debug=True, port=5000, host='localhost')
    app.run(host="0.0.0.0", port=5000)
