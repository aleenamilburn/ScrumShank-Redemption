import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_excel("MarchData.xlsx", sheet_name="cleaned")

# Extract fiscal year
df["Fiscal.Year"] = df["Ordered.Date"].dt.year

# Aggregate by fiscal year, category, and vendor
agg_data = df.groupby(["Fiscal.Year", "PO.Category.Description", "Vendor.Name"])['Line.Total'].sum().reset_index()

# Convert Fiscal Year to datetime format
agg_data["Fiscal.Year"] = pd.to_datetime(agg_data["Fiscal.Year"], format="%Y")

# Prepare time series for overall spending
overall_spending = df.groupby("Ordered.Date")['Line.Total'].sum().reset_index()
overall_spending = overall_spending.rename(columns={"Ordered.Date": "ds", "Line.Total": "y"})

# Prophet Forecasting
prophet_model = Prophet()
prophet_model.fit(overall_spending)
future = prophet_model.make_future_dataframe(periods=3 * 365, freq="D")
forecast = prophet_model.predict(future)
prophet_model.plot(forecast)
plt.show()

# ARIMA Forecasting
arima_data = overall_spending.set_index("ds")["y"]
model = ARIMA(arima_data, order=(5,1,0))
model_fit = model.fit()
forecast_arima = model_fit.forecast(steps=3 * 365)
plt.figure(figsize=(10,5))
plt.plot(arima_data, label="Historical")
plt.plot(pd.date_range(start=arima_data.index[-1], periods=len(forecast_arima), freq="D"), forecast_arima, label="Forecast", color="red")
plt.legend()
plt.show()

# LSTM Forecasting
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(overall_spending[['y']])
SEQ_LENGTH = 30

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

future_inputs = scaled_data[-SEQ_LENGTH:]
future_preds = []
for _ in range(365):
    pred = model.predict(future_inputs.reshape(1, SEQ_LENGTH, 1))
    future_preds.append(pred[0][0])
    future_inputs = np.append(future_inputs[1:], pred, axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
plt.figure(figsize=(10,5))
plt.plot(overall_spending["ds"], overall_spending["y"], label="Historical")
plt.plot(pd.date_range(start=overall_spending["ds"].iloc[-1], periods=len(future_preds), freq="D"), future_preds, label="Forecast", color="green")
plt.legend()
plt.show()

# Export results for Tableau
forecast.to_csv("forecast_results.csv", index=False)
