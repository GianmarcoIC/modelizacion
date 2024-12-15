# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from graphviz import Digraph

# Datos simulados para entrenamiento y prueba
np.random.seed(42)
years = np.arange(2000, 2025)
articles = np.random.poisson(lam=20, size=len(years))  # Datos simulados

data_simulated = pd.DataFrame({"anio_publicacion": years, "cantidad_articulos": articles})

# Preparar datos para los modelos
X = data_simulated[["anio_publicacion"]]
y = data_simulated["cantidad_articulos"]
X_normalized = (X - X.min()) / (X.max() - X.min())
y_normalized = (y - y.min()) / (y.max() - y.min())

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Años para predicción
anos_prediccion = np.arange(2025, 2031)
anos_normalizados = (pd.DataFrame(anos_prediccion) - X.min().values[0]) / (X.max().values[0] - X.min().values[0])

# Modelo de Red Neuronal
modelo_nn = Sequential([
    Dense(5, activation='relu', input_dim=1),
    Dense(5, activation='relu'),
    Dense(1, activation='linear')
])
modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)

# Predicciones de Red Neuronal
predicciones_nn = modelo_nn.predict(anos_normalizados)
predicciones_desnormalizadas = predicciones_nn * (y.max() - y.min()) + y.min()

# Modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.values, y_train.values)

# Predicciones de Random Forest
pred_rf = rf_model.predict(anos_normalizados.values)
pred_rf_desnormalizadas = pred_rf * (y.max() - y.min()) + y.min()

# Crear tabla comparativa
comparison_df = pd.DataFrame({
    "Año": anos_prediccion,
    "Red Neuronal (Predicciones)": predicciones_desnormalizadas.flatten(),
    "Random Forest (Predicciones)": pred_rf_desnormalizadas
})

# Calcular métricas
mse_nn = mean_squared_error(y_test.values, modelo_nn.predict(X_test))
mse_rf = mean_squared_error(y_test.values, rf_model.predict(X_test.values))
comparison_df["Diferencia Absoluta"] = abs(comparison_df["Red Neuronal (Predicciones)"] - comparison_df["Random Forest (Predicciones)"])

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(anos_prediccion, comparison_df["Red Neuronal (Predicciones)"], label="Red Neuronal", marker='o', color='blue')
plt.plot(anos_prediccion, comparison_df["Random Forest (Predicciones)"], label="Random Forest", marker='s', color='green')
plt.title("Comparación de Predicciones: Red Neuronal vs Random Forest")
plt.xlabel("Año")
plt.ylabel("Cantidad de Artículos")
plt.legend()
plt.grid()
plt.show()

# Generar gráfico de red neuronal
nn_graph = Digraph(format="png")
nn_graph.attr(rankdir="LR")

# Capa de entrada
nn_graph.node("Input", "Año", shape="circle", style="filled", color="lightblue")

# Primera capa oculta
for i in range(1, 6):
    nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
    nn_graph.edge("Input", f"Hidden1_{i}")

# Segunda capa oculta
for i in range(1, 6):
    for j in range(1, 6):
        nn_graph.node(f"Hidden2_{j}", f"Oculta 2-{j}", shape="circle", style="filled", color="lightgreen")
        nn_graph.edge(f"Hidden1_{i}", f"Hidden2_{j}")

# Capa de salida
nn_graph.node("Output", "Predicción", shape="circle", style="filled", color="orange")
for i in range(1, 6):
    nn_graph.edge(f"Hidden2_{i}", "Output")

nn_graph.render("nn_graph", view=True)

# Imprimir tabla comparativa y métricas
print("\nTabla Comparativa:\n", comparison_df)
print("\nError Cuadrático Medio (MSE):")
print(f"Red Neuronal: {mse_nn:.4f}")
print(f"Random Forest: {mse_rf:.4f}")
