import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Cargar los datos
data = pd.read_csv('btc_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Obtener los precios de cierre diarios
prices = data['Close']

# Calcular la volatilidad
data['Volatility'] = data['High'] - data['Low']

# Calcular el volumen promedio semanal
volume_weekly = data['Volume'].resample('W').mean()

# Convertir los precios y el volumen a precios semanales
prices_weekly = prices.resample('W').last()
data_weekly = pd.concat(
    [prices_weekly, volume_weekly, data['Volatility']], axis=1)

# Crear el modelo de suavizado exponencial simple
model = SimpleExpSmoothing(data_weekly).fit()

# Pronosticar los precios de cierre para las próximas 100 semanas
data_pred = data_weekly.append(pd.DataFrame(index=pd.date_range(
    start=data_weekly.index[-1], periods=100, freq='W')))
data_pred['Close'] = model.forecast(len(data_pred))

# Imprimir los precios de cierre pronosticados en forma de tabla
print(data_pred['Close'].resample('W').last().to_string())
