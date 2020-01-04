import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
import pandas as pd 

import quandl
quandl.ApiConfig.api_key = "5UDsQj-5EYJNHoo8UbKu"

df = quandl.get("WIKI/AMZN", start_date="2002-01-01", end_date="2002-01-30")
df.reset_index(inplace=True)
print(df)

# Get data function
def get_data(df):  
    data = df.copy()
    data['Date'] = data['Date'].astype(str).str.split('-').str[2]
    data['Date'] = pd.to_numeric(data['Date'])
    data['Close'] = pd.to_numeric(data['Close'])
    print(data['Date'])
    return [ data['Date'].tolist(), data['Close'].tolist() ] 
    
# Convert Series to listdates    
dates, prices = get_data(df)
# print(prices)

# predict and plot function
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1))
    # convert to 1xn dimension
    x = np.reshape(x,(len(x), 1))
    
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Fit regression model
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x), svr_lin.predict(x), svr_poly.predict(x)

predicted_price = predict_prices(dates, prices, [31])
print(predicted_price)