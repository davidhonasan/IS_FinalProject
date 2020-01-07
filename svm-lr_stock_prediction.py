import quandl
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from predict_xgboost import stock_predict

# FB, AAPL, DIS
quandl.ApiConfig.api_key = "EMUFhFWXZLfbkxVVx_Tx"

def main(stock):

	# Get the stock data
	df = quandl.get(stock)

	# Get the Adjusted Close Price 
	df = df[['Adj. Close']] 
	print(df)

	# A variable for predicting 'n' days out into the future
	forecast_out = 30 #'n=30' days
	#Create another column (the target ) shifted 'n' units up
	df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

	### Create the independent data set (X)  #######
	# Convert the dataframe to a numpy array
	X = np.array(df.drop(['Prediction'],1))

	#Remove the last '30' rows
	X = X[:-forecast_out]

	### Create the dependent data set (y)  #####
	# Convert the dataframe to a numpy array 
	y = np.array(df['Prediction'])

	# Get all of the y values except the last '30' rows
	y = y[:-forecast_out]

	# Split the data into 80% training and 20% testing
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# Create and train the Support Vector Machine (Regressor)  
	# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ (Kernel type) rbf is the default one
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
	svr_rbf.fit(x_train, y_train)

	# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
	# The best possible score is 1.0
	svm_confidence = svr_rbf.score(x_test, y_test)
	print("svm confidence: ", svm_confidence)

	# Create and train the Linear Regression  Model
	lr = LinearRegression()
	# Train the model
	lr.fit(x_train, y_train)

	# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
	# The best possible score is 1.0
	lr_confidence = lr.score(x_test, y_test)
	print("lr confidence: ", lr_confidence)

	# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
	x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
	# print(x_forecast)

	# Print linear regression model predictions for the next '30' days
	lr_prediction = lr.predict(x_forecast)
	# print(lr_prediction)

	# Print support vector regressor model predictions for the next '30' days
	svm_prediction = svr_rbf.predict(x_forecast)
	# print(svm_prediction)


	# Root mean square error check
	rms_lr = sqrt(mean_squared_error(x_forecast, lr_prediction))
	rms_svm = sqrt(mean_squared_error(x_forecast, svm_prediction))

	print("RMSE lr  :", rms_lr)
	print("RMSE SVM :", rms_svm)

	result = [x_forecast, lr_prediction, svm_prediction,rms_lr,rms_svm]
	return result

	# #Plot the data
	# plt.plot(x_forecast, marker="o", label="Real")
	# plt.plot(lr_prediction, label="lr")
	# plt.plot(svm_prediction, label="svm")
	# plt.legend()
	# plt.show()

fb = main("WIKI/FB")
apple = main("WIKI/AAPL")
dis = main("WIKI/DIS")

print(dis[2])

rmse_lr = [fb[3], apple[3], dis[3]]
rmse_svm = [fb[4], apple[4], dis[4]]
xgboost = [stock_predict("FB"), stock_predict("AAPL"), stock_predict("DIS")]
company = ["Facebook", "Apple", "Disney"]

#Plotting 
fig, axs = plt.subplots(2, 2, figsize=(13,9))
axs[0, 0].plot(fb[0], label="Real", marker="o")
axs[0, 0].plot(fb[1], label="lr")
axs[0, 0].plot(fb[2], label='svm')
axs[0, 0].legend()
axs[0, 0].set_title('Facebook')
axs[0, 1].plot(apple[0], label="Real", marker="o")
axs[0, 1].plot(apple[1], label="lr")
axs[0, 1].plot(apple[2], label="lr")
axs[0, 1].legend()
axs[0, 1].set_title('Apple')
axs[1, 0].plot(dis[0], label="Real", marker="o")
axs[1, 0].plot(dis[1], label="lr")
axs[1, 0].plot(dis[2], label="svm")
axs[1, 0].legend()
axs[1, 0].set_title('Disney')
axs[1, 1].plot(company, rmse_lr, label="rmse_lr")
axs[1, 1].plot(company, rmse_svm, label="rmse_svm")
axs[1, 1].plot(company, xgboost, label="rmse_xgboost")
axs[1, 1].legend()
axs[1, 1].set_title('RMSE')


plt.show()
# main("WIKI/DIS")