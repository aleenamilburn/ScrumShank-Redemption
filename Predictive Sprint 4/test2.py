from fbprophet import Prophet

# Prepare data for overall spending trend
overall_spending = df.groupby("Ordered.Date")["Line.Total"].sum().reset_index()
overall_spending = overall_spending.rename(columns={"Ordered.Date": "ds", "Line.Total": "y"})

# Initialize and fit the model
prophet_model = Prophet()
prophet_model.fit(overall_spending)

# Create future dataframe for three years ahead
future = prophet_model.make_future_dataframe(periods=3 * 365, freq="D")
forecast = prophet_model.predict(future)

# Plot results
prophet_model.plot(forecast)
plt.show()
