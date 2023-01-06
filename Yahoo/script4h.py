# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

# %%
br = yf.download(tickers='BRLUSD=X', period='1y', interval='1h')
eur = yf.download(tickers='EURUSD=X', period='1y', interval='1h')
xrp = yf.download(tickers='XRP-USD', period='1y', interval='1h')
xlm = yf.download(tickers='XLM-USD', period='1y', interval='1h')
solana = yf.download(tickers='SOL-USD', period='1y', interval='1h')
btc = yf.download(tickers='BTC-USD', period='1y', interval='1h')
silver = yf.download(tickers='SI=F', period='1y', interval='1h')
gold = yf.download(tickers='GC=F', period='1y', interval='1h')
libra = yf.download(tickers='GBPUSD=X', period='1y', interval='1h')
eth = yf.download(tickers='ETH-USD', period='1y', interval='1h')
oil = yf.download(tickers='CL=F', period='1y', interval='1h')

# %%
master = pd.concat([br['Close'], eur['Close'], xrp['Close'], xlm['Close'], solana['Close'], btc['Close'], silver['Close'], gold['Close'], libra['Close'], eth['Close'] , oil['Close']], axis=1)
master.columns = ['br', 'eur', 'xrp', 'xlm', 'solana', 'btc', 'silver', 'gold', 'libra', 'eth', 'oil']

# %%
# Conversion to 4hrs
data = master
data_4h = data[(data.index.astype('str').str.slice(11,13).astype('int') == 1) | (data.index.astype('str').str.slice(11,13).astype('int') == 5)
    | (data.index.astype('str').str.slice(11,13).astype('int') == 9) | (data.index.astype('str').str.slice(11,13).astype('int') == 13) |
    (data.index.astype('str').str.slice(11,13).astype('int') == 17) | (data.index.astype('str').str.slice(11,13).astype('int') == 21)]

# %%
sns.heatmap(data_4h.isna())

# %%
#Dataset split to maximize information in predictions

#For stocks with full information, use just stocks with full information
features_full = data_4h.drop(['br', 'eur', 'silver', 'gold', 'libra', 'oil'], axis=1)
features_full

# %%
#stationary = data_4h.pct_change(1)
stationary = features_full
series_names = stationary.columns

lag_1 = stationary.shift(1)
lag_2 = stationary.shift(2)
lag_3 = stationary.shift(3)
lag_4 = stationary.shift(4)
lag_5 = stationary.shift(5)
lag_6 = stationary.shift(6)

for i in range(1,7):
    exec(f"lag_{i}.columns = series_names + '_lag{i}'")
    
final_full = pd.concat([stationary, lag_1, lag_2, lag_3, lag_4, lag_5, lag_6], axis=1)
final_full.dropna(axis=0, inplace=True)

# %%
y = final_full["xrp"] #
x = final_full.iloc[:,5:] #past data

# %%
import xgboost as xgb
model = xgb.XGBRegressor(n_jobs=1, tree_method='hist', random_state=0, max_bin=500)
model.fit(x,y)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y.values[-30:], label='Valores reales')
plt.plot(model.predict(x)[-30:], label='Pronóstico')
plt.scatter(x=30,
            y=model.predict(pd.concat([final_full.iloc[-1:,0:5], x.iloc[-1:,0:-5]], axis=1).values),
           color='green', label='Pronostico Futuro')
plt.title(str(y.name))
plt.legend()

# %%
print('------------------------------------------------------------------')
print('Forecast serie:', y.name)
print('------------------------------------------------------------------')

print('Actual data:', final_full.index[-1])

print(data_4h['xrp'].iloc[-1])

#print('Past forecast:', model.predict(x.iloc[-1:,:]).item())

print('------------------------------------------------------------------')
print('Forecast data (4 hours ahead):', pd.Timestamp(data_4h.index[-1].asm8 + int(1.44e13)))

print('Predicted change(%):', str(np.round(((model.predict(pd.concat([final_full.iloc[-1:,0:5], x.iloc[-1:,0:-5]], axis=1).values) -
                               data_4h['xrp'].iloc[-1])/data_4h['xrp'].iloc[-1]).item()*100, 2))+'%')
print('------------------------------------------------------------------')

print('4 hours ahead forecast:')
print(model.predict(pd.concat([final_full.iloc[-1:,0:5], x.iloc[-1:,0:-5]], axis=1).values).item())

fc_xrp = model.predict(pd.concat([final_full.iloc[-1:,0:5], x.iloc[-1:,0:-5]], axis=1).values).item()

# %%
#For stocks with incomplete information, GBP and Silver, Oil, all the dataset will be used
features_full = data_4h.drop(['br', 'silver', 'gold', 'oil'], axis=1)
features_full
#stationary = data_4h.pct_change(1)
stationary = features_full
series_names = stationary.columns

lag_1 = stationary.shift(1)
lag_2 = stationary.shift(2)
lag_3 = stationary.shift(3)
lag_4 = stationary.shift(4)
lag_5 = stationary.shift(5)
lag_6 = stationary.shift(6)

for i in range(1,7):
    exec(f"lag_{i}.columns = series_names + '_lag{i}'")
    
final_full = pd.concat([stationary, lag_1, lag_2, lag_3, lag_4, lag_5, lag_6], axis=1)
final_full.dropna(axis=0, inplace=True)

# %%
y = final_full['libra'] #
x = final_full.iloc[:,7:] #past data

# %%
import xgboost as xgb
model = xgb.XGBRegressor(n_jobs=1, tree_method='hist', random_state=0, max_bin=500)
model.fit(x,y)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y.values[-74:], label='Valores reales')
plt.plot(model.predict(x)[-74:], label='Pronóstico')
plt.scatter(x=74,
            y=model.predict(pd.concat([final_full.iloc[-1:,0:7], x.iloc[-1:,0:-7]], axis=1).values),
           color='green', label='Pronostico Futuro')
plt.title(str(y.name))
plt.legend()

# %%
print('------------------------------------------------------------------')
print('Forecast serie:', y.name)
print('------------------------------------------------------------------')

print('Actual data:', final_full.index[-1])

print(data_4h['libra'].iloc[-1])

#print('Past forecast:', model.predict(x.iloc[-1:,:]).item())

print('------------------------------------------------------------------')
print('Forecast data (4 hours ahead):', pd.Timestamp(data_4h.index[-1].asm8 + int(1.44e13)))

print('Predicted change(%):', str(np.round(((model.predict(pd.concat([final_full.iloc[-1:,0:7], x.iloc[-1:,0:-7]], axis=1).values) -
                               data_4h['libra'].iloc[-1])/data_4h['libra'].iloc[-1]).item()*100, 2))+'%')
print('------------------------------------------------------------------')

print('4 hours ahead forecast:')
print(model.predict(pd.concat([final_full.iloc[-1:,0:7], x.iloc[-1:,0:-7]], axis=1).values).item())

fc_libra = model.predict(pd.concat([final_full.iloc[-1:,0:7], x.iloc[-1:,0:-7]], axis=1).values).item()

# %%
# Write with every run
print('Currently writing file:', str(int(data_4h.index[-1].asm8 + int(1.44e13))/int(1.44e13) - 116171.25))
pd.DataFrame([[fc_xrp, fc_libra]], columns=['fc_xrp', 'fc_libra'], index=[pd.Timestamp(data_4h.index[-1].asm8 + int(1.44e13))]).to_csv("fc_" + str(int(data_4h.index[-1].asm8 + int(1.44e13))/int(1.44e13) - 116171.25) + ".csv")


