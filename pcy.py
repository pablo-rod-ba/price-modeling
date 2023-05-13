import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Cargar los datos
data = pd.read_csv('btc_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Obtener los precios de cierre diarios
prices = data['Close']

# Convertir los precios a precios semanales
prices_weekly = prices.resample('W').last()

# Ajustar el modelo ARIMA
model = ARIMA(prices_weekly, order=(1, 1, 1))
model_fit = model.fit()

# Pronosticar los precios de cierre para las próximas 100 semanas
forecast = model_fit.forecast(steps=100)

# Convertir los precios de cierre pronosticados de vuelta a precios diarios
dates_pred = pd.date_range(
    start=data.index[-1], periods=len(forecast), freq='D')[1:]
prices_pred = pd.Series(forecast, index=dates_pred)

# Imprimir los precios de cierre pronosticados en forma de tabla
print("Precios pronosticados:")
print(prices_pred.resample('W').last().to_string())

# Actualizar los precios con los precios pronosticados
prices = pd.concat([prices, prices_pred])

# Imprimir los precios de cierre actualizados
print("\nPrecios actualizados:")
print(prices.resample('W').last().to_string())
