# Simple linear regression analysis of stock prices using quandl, pandas and sklearn
# Sourabh Kulkarni
import pandas as pd 
import quandl, math, datetime
import numpy as np 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')

# Import Google stock price data from quandl
df = quandl.get('WIKI/GOOGL', api_key = 'yad_xM6acsSsR6ZXzhC5')

# Select the data features we are interested in
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Create new features which provide more insight
# High-Low Percentage - guage the volatility of the stock
df['HL_PCT'] = 100 * (df['Adj. High'] - df['Adj. Low']) / (df['Adj. Low'])
# Percent change daily
df['PCT_change'] = 100 * (df['Adj. Close'] - df['Adj. Open']) / (df['Adj. Open'])

df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]

# Now we build forecast. The code from now onwards can be used for any kind of data. 
forecast_col = 'Adj. Close'
# Replace NA values with a number 
df.fillna(-99999, inplace=True)
# How far into future do you want to predict? 0.01 = 1%
forecast_window = 0.1
forecast_out = int(math.ceil(forecast_window*len(df)))
# The forecast is close values of future days, push them back by forecast_out amount and now they are the labels we would like to forecast!
df['label'] = df[forecast_col].shift(-forecast_out)


# Seperate dataset into features and labels, normalize data.
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])
#y = preprocessing.scale(y)
# Split dataset 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

# pickle_in = open('linearregression.pickle','rb')

# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

print (accuracy)

# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)

# print (accuracy)

forecast_set = clf.predict(X_lately)

print (forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail(10))

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()