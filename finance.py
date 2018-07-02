import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle # serialization - arrange something in a series

# regression - takes continous data and find best fit
# features or labels?
# features are attributes of what may cause adjusted price (in 10 days or 10%)
# price is label
# X is features, y is labels

g_df = quandl.get('EOD/GOOGL') # Google stock, EOD sample data
df = pd.DataFrame(g_df)

# print(df.head()) 
# prints date, open, high, low, close, volume, ex-divident,
# split ratio, adj.open, adj.high, adj.low, adj.close, adj.volume 

# split shares denoted by 'Adj.' - more accurate measure of true stock price
# high-low shows margin of volatility for the day
# open-close shows change in price (up/down) within one day

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]          
# reflect split stock - more accurate 

df['HL%'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# high-low percentage change, percent volatility

df['%CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# daily percent change

# only thing affecting price is Adj. Close, must drop it
df = df[['Adj. Close','HL%','%CHANGE','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) 
# treated as an outlier, but does not sacrifice data by deleting column
forecast_out = int(math.ceil(0.1*len(df))) # prints 30 days in advanced 
# rounds everything to the nearest whole number (integer form)
# predicts out 10% of the DataFrame, using 10 days ago data to predict today
df['label'] = df[forecast_col].shift(-forecast_out) # prediction column
# shift column negatively, shifted up - label column for each row will be 
# adjusted close price 10 days into the future

X = np.array(df.drop(['label', 'Adj. Close'],1)) 
# returns new DataFrame converted to numpy array stored as X
X = preprocessing.scale(X) # normalize 
X_recent = X[-forecast_out:] # predict agasint, find m and b
# don't have y values, not trained or tested agasint this data
X = X[:-forecast_out]
df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# testing 20% of data, take all features and labels, shuffles, fit classifiers
clf = LinearRegression(n_jobs=-1) # running as many possible threads/jobs at once
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
# saving classifier to avoid training step, dumps classifier 

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in) # don;t need to train classifier every time

accuracy = clf.score(X_test, y_test) # prints 0.967, squared error
# train and test on seperate data 

# predict based on X data
forecast_set = clf.predict(X_recent)
# passes single/array of values, output in same order the values
# each investment/price report is a day, means each forecast is a day later
print(forecast_set, accuracy, forecast_out) # next 30 days of stock prices
df['Forecast'] = np.nan

last_date = df.iloc[-1].name # last date and name
last_unix = last_date.timestamp() # last unix value
one_day = 86400 #seconds in a day
next_unix = last_unix + one_day # next day
 
for i in forecast_set: # have days on axis
# iterating through forecast set taking each forecast and day 
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
# setting those as the values in DataFrame, making future features NaN
    df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)] + [i]
# references index (date), next date is a datestamp
# if index exist, replace. if index does not exist, create it
# all future values but Forecast are NaN (do not have that data yet)
 
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()