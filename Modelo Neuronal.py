# ================================
# PARTE 1: PREPARACION DE DATOS
# ================================
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

#Cargar Archivo
df1 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cordoba1")
df2 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira2")
df3 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira3")
df4 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira4")
df5 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Antioquia5")
df6 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Antioquia6")
df7 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Atlantico7")
df8 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Atlantico8")
df9 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Magdalena9")
df10 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Magdalena10")
df11 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cesar11")
df12 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cesar12")
df13 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Bolivar13")
df14 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Bolivar14")
df15 = pd.read_excel("Datos sin irrigacion.xlsx", sheet_name = "Resumen_Choco15")

# Lista Df
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]

# Unirlos en uno solo
df = pd.concat(dfs, ignore_index=True)

# Dividir en X e y
X = df.drop("Toneladas por hectaria", axis=1)
y = df["Toneladas por hectaria"]

# Train-test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacion
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(train_X.to_numpy())  



# ================================
# PARTE 2: BÚSQUEDA MANUAL DE HIPERPARÁMETROS
# ================================
activations = ['relu', 'tanh']
optimizers = ['adam', 'sgd']
learning_rates = [0.001, 0.01]
losses = ['mean_absolute_error', 'mean_squared_error']
num_layers_list = [1, 2]
units_list = [64, 128]

results = []

for activation in activations:
    for optimizer_name in optimizers:
        for learning_rate in learning_rates:
            for loss in losses:
                for num_layers in num_layers_list:
                    for units in units_list:

                        model = tf.keras.Sequential()
                        model.add(norm_layer)

                        for _ in range(num_layers):
                            model.add(tf.keras.layers.Dense(units=units, activation=activation))

                        model.add(tf.keras.layers.Dense(1))

                        if optimizer_name == 'adam':
                            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                        else:
                            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

                        history = model.fit(train_X, train_y, validation_split=0.2, epochs=50, verbose=0)

                        y_pred = model.predict(test_X).flatten()

                        if np.isnan(y_pred).any():
                            continue

                        mse = mean_squared_error(test_y, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(test_y, y_pred)
                        r2 = r2_score(test_y, y_pred)

                        results.append({
                            'activation': activation,
                            'optimizer': optimizer_name,
                            'learning_rate': learning_rate,
                            'loss': loss,
                            'num_layers': num_layers,
                            'units': units,
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2
                        })

# Convertir resultados en DataFrame
results_df = pd.DataFrame(results)

# Mostrar el mejor modelo
best_model = results_df.sort_values('MAE').iloc[0]
print("\nMejores hiperparámetros encontrados:")
print(best_model)


# ================================
# PARTE 3: GRAFICOS
# ================================
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Loss entrenamiento')
plt.plot(history.history['val_loss'], label='Loss validacion')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje')
plt.grid(True)
plt.show()