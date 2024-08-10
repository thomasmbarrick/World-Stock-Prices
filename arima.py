# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and prepare dataset
df = pd.read_csv("World-Stock-Prices-Dataset.csv")

# Filter for the desired brand
df = df[df["Brand_Name"] == "peloton"]

# Drop unnecessary columns
df = df.drop(columns=['Capital Gains', 'Country', 'Industry_Tag', 'Ticker', 'Volume', 'Dividends', 'Stock Splits', 'Low', 'High', 'Open'])

# Ensure the 'Close' column is numeric and handle missing values
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# Convert 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select the relevant time series
series = df['Close']

# Fit ARIMA model
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()

# Summary of the fit model
print(model_fit.summary())

# Plot residuals
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(12, 6))

# Line plot of residuals
plt.subplot(1, 2, 1)
plt.plot(residuals)
plt.title('Residuals')

# Density plot of residuals
plt.subplot(1, 2, 2)
residuals.plot(kind='kde', ax=plt.gca())
plt.title('Density Plot of Residuals')

plt.tight_layout()
plt.show()

# Summary statistics of residuals
print(residuals.describe())

