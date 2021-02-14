#%% [markdown]
# Get some idea for the dataset

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.dates as mdates
green = sns.color_palette('deep', 8)[2]
blue = sns.color_palette('deep', 8)[0]
import datetime

# Weather Data
df_weather = pd.read_csv("weatherHistory.csv")
#Cleaning the dates column
df_weather['Formatted Date'] = pd.to_datetime(df_weather['Formatted Date'])
#Plotting the raw weather data
fig = plt.figure(figsize=(17,8))
ax1 = fig.add_subplot(121)
plt.scatter(df_weather["Formatted Date"],df_weather["Temperature (C)"], color=green,s=20)
plt.title("Weather Data Time Series",fontsize=15)
plt.xlabel("Date",fontsize=15)
plt.ylabel("Temperature (ÂºC)",fontsize=15)
# Plotting the autocorrelation plot
ax2 = fig.add_subplot(122)
plot_acf(df_weather["Temperature (C)"], ax=ax2,color=blue)
plt.title("Autocorrelation Plot for Weather Data", fontsize=15)
plt.ylabel("Correlation",fontsize=15)
plt.xlabel("Lag",fontsize=15)
plt.show()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.dates as mdates
green = sns.color_palette('deep', 8)[2]
blue = sns.color_palette('deep', 8)[0]
import datetime

retail_sales = 'sales_data.csv'
df_sales = pd.read_csv(retail_sales)
fig = plt.figure(figsize=(17,8))
ax1 = fig.add_subplot(121)
df_sales_sum = df_sales.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df_sales.sum['Date'] = pd.to_datetime(df_sales_sum['Date'])
df_sales.plot(x='Date', y='Weekly_Sales', color='g', ax=ax1, fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.title('Total Sales Volume', fontsize=15)
plt.ylabel('Sales', fontsize=15)

date_form = mdates.DateFormatter('%Y-%m-%d')
year_locator = mdates.YearLocator()
ax1.xaxis.set_major_locator(year_locator)
ax2 = fig.add_subplot(122)
plot_acf(df_sales_sum.Weekly_Sales, ax=ax2)
plt.title('Autocorrelation', fontsize=15)
plt.xlabel('Lag', fontsize=15)
plt.ylabel('Correlation', fontsize=15)
plt.show()

#%% [markdown]
# Check Stationary
#%%
from statsmodels.tsa.stattools import adfuller

adf_test_sales = adfuller(list(df_sales_sum['Weekly_Sales']))
adf_test_weather = adfuller(list(df_weather['Temporature (C)']))
print('Weather Results:')
print('ADF = ' + str(adf_test_weather[0]))
print('p-value = ' + str(adf_test_weather[1]))

print('Retail Results:')
print('ADF = ' + str(adf_test_sales[0]))
print('p-value = ' + str(adf_test_sales[1]))

#%% [markdown]
# Check Trends
#%%
from fbprophet import Prophet
from datetime import datetime

start_date = "2007-01-01"
end_date = "2008-12-31"
df_weather["Formatted Date"] = pd.to_datetime(df_weather["Formatted Date"], utc=True)

date_range = (df_weather["Formatted Date"] > start_date) & (df_weather["Formatted Date"] < end_date)

df_prophet = df_weather.loc[date_range]

m = Prophet()


ds = df_prophet["Formatted Date"].dt.tz_localize(None)
y = df_prophet["Temperature (C)"]
df_for_prophet = pd.DataFrame(dict(ds=ds,y=y))
m.fit(df_for_prophet)

future = m.make_future_dataframe(periods=120)


forecast = m.predict(future)
forecast = forecast[["ds","trend", "trend_lower", "trend_upper"]]
fig = m.plot_components(forecast,plot_cap=False)
trend_ax = fig.axes[0]


trend_ax.plot()
plt.title("Trend for Weather Data", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.ylabel("Weather Trend", fontsize=15)
plt.show()

#%%
from fbprophet import Prophet


m = Prophet()
# Selecting one store
df_store_1 = df_sales[df_sales["Store"]==1]

df_store_1["Date"] = pd.to_datetime(df_store_1["Date"])
ds = df_store_1["Date"].dt.tz_localize(None)
y = df_store_1["Weekly_Sales"]
df_for_prophet = pd.DataFrame(dict(ds=ds,y=y))
m.fit(df_for_prophet)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)
forecast = forecast[["ds","trend", "trend_lower", "trend_upper"]]
fig = m.plot_components(forecast,plot_cap=False)
trend_ax = fig.axes[0]
trend_ax.plot()
plt.title("Trend for Retail Data", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.ylabel("Sales Trend", fontsize=15)

plt.show()

#%% [markdown]
# Moving Average
#%%
from sklearn.metrics import mean_absolute_error
import numpy as np

green = sns.color_palette('deep', 8)[2]
blue = sns.color_palette('deep', 8)[0]

start_date = "2007-01-01"
end_date = "2008-12-31"
df_weather["Formatted Date"] = pd.to_datetime(df_weather["Formatted Date"], utc=True)
date_range = (df_weather["Formatted Date"] > start_date) & (df_weather["Formatted Date"] < end_date)
df_weather_ma = df_weather.loc[date_range]
series = df_weather_ma["Temperature (C)"]

window = 90
rolling_mean = series.rolling(window=window).mean()
fig, ax = plt.subplots(figsize=(17,8))
plt.title('Moving Average Model for Weather Dataset', fontsize=15)
plt.plot(rolling_mean, color=green, lable='Rolling mean trend')
# plot confidence intervals for smoothed values
mae = mean_absolute_error(series[window:], rolling_mean[window:])
deviation = np.std(series[window:] - rolling_mean[window:])
lower_bound = rolling_mean - (mae + 2 * deviation)
upper_bound = rolling_mean + (mae + 2 * deviation)

plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
plt.plot(lower_bound, 'r--')
plt.plot(series[window:], color=blue, label='Actual Values')
plt.legend(loc='best')
plt.grid(True)
plt.xticks([])
plt.show()

# Retail
series = df_sales_sum.Weekly_Sales
window=15

rolling_mean = series.rolling(window=window).mean()
fig,ax = plt.subplots(figsize=(17,8))


plt.title('Moving Average Model for Retail Sales',fontsize=15)

plt.plot(rolling_mean, color=green, label='Rolling mean trend')

#Plot confidence intervals for smoothed values
mae = mean_absolute_error(series[window:], rolling_mean[window:])
deviation = np.std(series[window:] - rolling_mean[window:])
lower_bound = rolling_mean - (mae + 1.92 * deviation)
upper_bound = rolling_mean + (mae + 1.92 * deviation)

plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
plt.plot(lower_bound, 'r--')

plt.plot(series[window:], color=blue,label='Actual values')


plt.legend(loc='best')
plt.grid(True)
plt.xticks([])
plt.show()

#%% [markdown]
# Exponential Smoothing
#%%
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
import pandas as pd

fit1 = ExponentialSmoothing(df_weather['Temporature (C)'][0:200]).fit(smoothing_level=0.1, optimized=False)
fit2 = ExponentialSmoothing(df_weather['Temporature (C)'][0:200]).fit(smoothing_level=0.5, optimized=False)
forecast1 = fit1.forecast(3).rename(r'$\alpha=0.1$')
forecast2 = fit2.forecast(3).rename(r'$\alpha=0.5$')

plt.figure(figsize=(17,8))
forecast1.plot(color='blue', legend=True)
forecast2.plot(color='red', legend=True)
df_weather['Temporature (C)'][0:200].plot(marker='', color='green', legend=True)
fit1.fittedvalues.plot(color='blue')
fit2.fittedvalues.plot(color='red')
plt.title('Exponential Smoothing for Weather Data', fontsize=15)
plt.xticks([])
plt.show()

# Retail

fit1 = ExponentialSmoothing(df_sales_sum["Weekly_Sales"][0:200]).fit(smoothing_level=0.1, optimized=False)

fit2 = ExponentialSmoothing(df_sales_sum["Weekly_Sales"][0:200]).fit(smoothing_level=0.5, optimized=False)

forecast1 = fit1.forecast(3).rename(r'$\alpha=0.1$')
forecast2 = fit2.forecast(3).rename(r'$\alpha=0.5$')
plt.figure(figsize=(17,8))

forecast1.plot(color='blue', legend=True)
forecast2.plot(color='red', legend=True)
df_sales_sum["Weekly_Sales"][0:200].plot(marker='',color='green', legend=True)
plt.ylabel("Sales", fontsize=15)

fit1.fittedvalues.plot(color='blue')
fit2.fittedvalues.plot(color='red')

plt.title("Exponential Smoothing for Retail Data", fontsize=15)
plt.xticks([], minor=True)
plt.show()

#%% [markdown]
# ARIMA
#%%
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

X = df_weather['Temporature (C)'].values
train_size = 600
test_size = 200
train, test = X[0:train_size], X[train_size:train_size+test_size]

history = [x for x in train]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

mse = mean_squared_error(test, predictions)
print(f'MSE error: {mse}')
plt.figure(figsize=(17,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('ARIMA fit Weather Data')
plt.xticks([])
plt.show()

# Sales dataset
X = df_sales_sum["Weekly_Sales"].values

split = int(0.66*len(X))
train, test = X[0:split], X[split:]

history = [x for x in train]
predictions = []
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
    
	obs = test[t]
	history.append(obs)
mse = mean_squared_error(test, predictions)

print(f"MSE error: {mse}")

plt.figure(figsize=(17,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.title("ARIMA fit to Sales Data",fontsize=15)
plt.xticks([])
plt.show()


