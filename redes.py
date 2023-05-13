import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Cargar los datos
data = pd.read_csv('btc_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Obtener los precios de cierre diarios
prices = data['Close']

# Convertir los precios a precios semanales
prices_weekly = prices.resample('W').last()

# Crear el conjunto de datos para la red neuronal
X = np.arange(len(prices_weekly)).reshape(-1, 1)
y = prices_weekly.values

# Crear la red neuronal
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1))
model
