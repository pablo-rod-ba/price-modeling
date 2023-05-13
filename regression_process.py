import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('btc_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Obtener los precios de cierre diarios
prices = data['Close']

# Convertir los precios a precios semanales
prices_weekly = prices.resample('W').last()

# Crear el conjunto de datos para la regresión
X = np.arange(len(prices_weekly)).reshape(-1, 1)
y = prices_weekly.values

# Crear el modelo de regresión lineal
reg = LinearRegression().fit(X, y)

# Pronosticar los precios de cierre para las próximas 100 semanas
X_pred = np.arange(len(prices_weekly), len(
    prices_weekly) + 100, 1).reshape(-1, 1)
y_pred = reg.predict(X_pred)

# Convertir los precios de cierre pronosticados de vuelta a precios diarios
dates_pred = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1), periods=len(y_pred), freq='D')
prices_pred = pd.Series(y_pred.flatten(), index=dates_pred)

# Imprimir los precios de cierre pronosticados en forma de tabla
print(prices_pred.resample('W').last().to_string())
