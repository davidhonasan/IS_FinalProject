from pandas_datareader import data
import yfinance as yf
yf.pdr_override() 

symbol = 'AMZN'
df = data.get_data_yahoo(symbol)
print(df)