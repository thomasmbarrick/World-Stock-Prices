# Import libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load and prepare dataset
df = pd.read_csv("World-Stock-Prices-Dataset.csv")

# Filter for the desired brand
df = df[df["Brand_Name"] == "peloton"]

# Drop unnecessary columns
df = df.drop(['Capital Gains', "Country", "Industry_Tag", "Ticker", "Volume", "Dividends", "Stock Splits", "Low", "High", "Open"], axis=1)

# Ensure the column of interest is numeric and handle missing values
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# Set the 'Date' column as the index if it's not already
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select the relevant time series
series = df['Close']

# Fit ARIMA model
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()

# Summary of fit model
print(model_fit.summary())

# Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('Residuals')
plt.show()

# Density plot of residuals
residuals.plot(kind='kde')
plt.title('Density Plot of Residuals')
plt.show()

# Summary stats of residuals
print(residuals.describe())
