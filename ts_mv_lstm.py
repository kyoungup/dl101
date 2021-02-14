# The source was not complete
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
import matplotlib.pyplot as plt
import numpy as np
from ml101.serialize import Stream
from ml101.preprocess import DataFilter


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# def difference(dataset, interval=1):
#     diff = list()
#     for i in range(interval, len(dataset)):
#         value = dataset[i] - dataset[i - interval]
#         diff.append(value)
#     return pd.Series(diff)


# def prepare_data(series, n_test, n_lag, n_seq):
#     raw_values = series.values
#     # transform data to be stationary
#     # diff_series = difference(raw_values, 1)
#     # diff_values = diff_series.values
#     # diff_values = diff_values.reshape(len(diff_values), 1)
#     # rescake values to -1,1
#     diff_values = raw_values
    
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_values = scaler.fit_transform(diff_values)
#     scaled_values = scaled_values.reshape(len(scaled_values), 1)
#     # transform into supervised learning problem X, y
#     supervised = series_to_supervised(scaled_values, n_lag, n_seq)
#     supervised_values = supervised.values
#     # split into train and test sets
#     train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
#     return scaler, train, test


# def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
#     X, y = train[:, 0:n_lag], train[:, n_lag:]
#     X = X.reshape(X.shape[0], 1, X.shape[1])

#     model = Sequential()
#     model.add(LSTM(n_neurons, batch_input_shape=(
#         n_batch, X.shape[1], X.shape[2]), stateful=True))
#     model.add(Dense(y.shape[1]))
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam', metrics=["accuracy"])
#     for i in range(nb_epoch):
#         model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
#         model.reset_states()
#     return model


# def forecast_lstm(model, X, n_batch):
#     X = X.reshape(1, 1, len(X))
#     forecast = model.predict(X, batch_size=n_batch)
#     return [x for x in forecast[0, :]]


# def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
#     forecasts = list()
#     for i in range(len(test)):
#         X, y = test[i, 0:n_lag], test[i, n_lag:]
#         forecast = forecast_lstm(model, X, n_batch)
#         forecasts.append(forecast)
#     return forecasts


# def inverse_difference(last_ob, forecast):
#     inverted = list()
#     inverted.append(forecast[0] + last_ob)
#     for i in range(1, len(forecast)):
#         inverted.append(forecast[i] + inverted[i-1])
#     return inverted


# def inverse_transform(series, forecasts, scaler, n_test):
#     inverted = list()
#     for i in range(len(forecasts)):
#         forecast = np.array(forecasts[i])
#         forecast = forecast.reshape(1, len(forecast))

#         inv_scale = scaler.inverse_transform(forecast)
#         inv_scale = inv_scale[0, :]

#         index = len(series) - n_test + i - 1
#         last_ob = series.values[index]
#         inv_diff = inverse_difference(last_ob, inv_scale)

#         inverted.append(inv_diff)
#     return inverted


# def evaluate_forecast(test, forecasts, n_lag, n_seq):
#     for i in range(n_seq):
#         actual = [row[i] for row in test]
#         predicted = [forecast[i] for forecast in forecasts]
#         rmse = math.sqrt(mean_squared_error(actual, predicted))
#         print('t+%d RMSE: %f' % ((i+1), rmse))


# def plot_forecasts(series, forecasts, n_test):
#     plt.plot(series.values)
#     for i in range(len(forecasts)):
#         off_s = len(series) - n_test + i - 1
#         off_e = off_s + len(forecasts[i]) + 1
#         xaxis = [x for x in range(off_s, off_e)]
#         yaxis = [series.values[off_s] + forecasts[i]]
#         plt.plot(xaxis, yaxis, color='red')
#     plt.show()


# configure
n_lag = 1
n_seq = 1
n_test = 8
n_epochs = 3    #30
n_batch = 28
n_neurons = 50

# load the dataset
data = pd.read_csv('pollution.csv', index_col=0)
values = data.values
# Encode Character Values
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1)) # dew and temp columns are negative
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, n_lag, n_seq)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
X, y = values[:, :n_test], values[:, -n_test]

# Splitting the dataset into the training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state = 0)
# reshape input to be 3D [sample, timesteps, features] i.e 8 features
train_X = train_X.reshape([train_X.shape[0], 1, train_X.shape[1]])
test_X = test_X.reshape([test_X.shape[0], 1, test_X.shape[1]])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# define the network
model = Sequential()
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
# fit the model
trained = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,
                    validation_data=(test_X, test_y), verbose=True, shuffle=False)
# plot history
plt.plot(trained.history['loss'], label='train')
plt.plot(trained.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print(f'Test RMSE: {rmse:.3f}')
