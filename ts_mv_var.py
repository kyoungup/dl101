# Multivariate time series forecasting _ by Mahbubul Alam _ Towards Data Science
# https://towardsdatascience.com/multivariate-time-series-forecasting-653372b3db36
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt


mdata = sm.datasets.macrodata.load_pandas().data
df = mdata.iloc[:, 2:4]
df.head()
plt.plot(df)
# plt.show()

# Test causality (data correlation)
# import for Granger's Causality Test
granger_test = sm.tsa.stattools.grangercausalitytests(
    df, maxlag=2, verbose=True)
# print(granger_test)

# Split data
nobs = 4
df_train, df_test = df[:-nobs], df[-nobs:]

# Check for stationary
def adf_test(ts, signif=0.5):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic',
                                        'p-value', '#Lags', '#Observatitions'])
    for key, value in dftest[4].items():
        adf[f'Critical Value ({key})'] = value
    print(adf)

    p = adf['p-value']

    if p <= signif:
        print(f'Series is Stationaty')
    else:
        print(f'Series is Non-Stationaty')
    return adf


# apply adf test on the series
adf = adf_test(df_train['realgdp'])
adf = adf_test(df_train['realcons'])

# Make data stationery
# 1st difference
df_differenced = df_train.diff().dropna()
# stationarity test again with differenced data
adf = adf_test(df_differenced['realgdp'])
adf = adf_test(df_differenced['realcons'])

# model fitting
model = VAR(df_differenced)
print(model.select_order().selected_orders)
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# Forecasting
lag_order = results.k_ar
results.forecast(df.values[-lag_order:], 5)
results.plot_forecast(20)
# plt.show()

# Evaluation
fevd = results.fevd(5)
print(fevd.summary())

# Inverting
pred = results.forecast(results.y, steps=nobs)
df_forecast = pd.DataFrame(
    pred, index=df.index[-nobs:], columns=df.columns + '_1d')
df_forecast.tail()

# inverting transformation
def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) +\
                df_fc[str(col)+'_1d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

# show inverted results in a dataframe
df_results = invert_transformation(df_train, df_forecast, second_diff=False)
print(df_results[['realgdp_forecast', 'realcons_forecast']])
