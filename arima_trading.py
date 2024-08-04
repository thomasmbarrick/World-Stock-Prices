import pandas as pd
from matplotlib import pyplot
df = pd.read_csv("World-Stock-Prices-Dataset.csv")

df = df[df["Brand_Name"] == "peloton"]
df = df.drop(['Capital Gains', "Country", "Industry_Tag","Ticker", "Volume", "Dividends", "Stock Splits", "Low", "High", "Open"], axis=1)
df.plot(x="Date", y="Close")
pyplot.show()