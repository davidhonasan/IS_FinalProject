# This program predicts stock prices by using machine learning models

#Install the dependencies
import quandl
quandl.ApiConfig.api_key = "5UDsQj-5EYJNHoo8UbKu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Get the stock data
df = quandl.get("WIKI/AMZN")
# Take a look at the data
# print(df.head())

# Get the Adjusted Close Price
df = df[['Adj. Close']]
#Take a look at the new data
# print(df.head())

# A variable for predicting 'n' days out into the future
forecast_out = 30 #'n=30' days
#Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
# df['Prediction'] = df['Adj. Close']
#print the new data set
# print(df.tail())

### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'],1))

#Remove the last 'n' rows
X = X[:-forecast_out]
# print(X)

### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]
# print(y)

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
# print("lr confidence: ", lr_confidence)

# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
# print("X_FORCEAST")
# print(x_forecast)

# Print linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
# print("LR_PREDICt")
# print(lr_prediction)

test = df.iloc[len(df)-30:len(df)]
print("test")
print(test)

result_predict = pd.DataFrame(lr_prediction)
# print(result_predict)
test['Prediction'] = result_predict[0].values
print(test)

plt.plot(test)
# ax = plt.plot(x_forecast)

# ax = df.plot(y='Adj. Close', style='b', grid=True, use_index=True)
# ax = df.plot(y='Prediction', style='r', grid=True, use_index=True, ax=ax)
plt.show()