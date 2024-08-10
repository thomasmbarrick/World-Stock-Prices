import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("World-Stock-Prices-Dataset.csv")

brand_name = input("Enter the brand name for analysis: ")

df = df[df["Brand_Name"] == brand_name]

if df.empty:
    print(f"No data found for brand '{brand_name}'.")
else:
    df = df.drop(columns=['Capital Gains', 'Country', 'Industry_Tag', 'Ticker', 'Volume', 'Dividends', 'Stock Splits', 'Low', 'High', 'Open'])

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    series = df['Close']

    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(residuals)
    plt.title('Residuals')

    plt.subplot(1, 2, 2)
    residuals.plot(kind='kde', ax=plt.gca())
    plt.title('Density Plot of Residuals')

    plt.tight_layout()
    plt.show()

    print(residuals.describe())


