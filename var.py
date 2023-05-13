import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# Cargar los datos
data = pd.read_csv('btc_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Obtener los precios de cierre diarios, así como Open, High, Low y Volume
prices = data['Close']
open_prices = data['Open']
high_prices = data['High']
low_prices = data['Low']
volume = data['Volume']

# Crear el conjunto de datos para el modelo VAR
data_var = pd.concat(
    [prices, open_prices, high_prices, low_prices, volume], axis=1)
data_var = data_var.dropna()

# Ajustar el modelo VAR
model = VAR(data_var)
results = model.fit()

# Pronosticar los precios de cierre para las próximas 100 semanas
lag_order = results.k_ar
forecast_input = data_var.values[-lag_order:]
fc = results.forecast(y=forecast_input, steps=100)

# Convertir los precios de cierre pronosticados de vuelta a precios diarios
dates_pred = pd.date_range(start=data.index[-1], periods=100, freq='W-SUN')
prices_pred = pd.Series(fc[:, 0], index=dates_pred)

# Imprimir los precios de cierre pronosticados en forma de tabla
print(prices_pred.to_string())
