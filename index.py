import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ..."

# Crear cliente Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Obtener datos
data = supabase.table("articulo").select("*").execute().data
df = pd.DataFrame(data)
df['anio_publicacion'] = pd.to_numeric(df['anio_publicacion'], errors="coerce")
datos_modelo = df.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

X = datos_modelo[['anio_publicacion']]
y = datos_modelo['cantidad_articulos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Red Neuronal
modelo_nn = Sequential([
    Dense(5, activation='relu', input_dim=1),
    Dense(5, activation='relu'),
    Dense(1, activation='linear')
])
modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)
predicciones_nn = modelo_nn.predict(X_test)
error_nn = mean_squared_error(y_test, predicciones_nn)

# Modelo Random Forest
modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)
predicciones_rf = modelo_rf.predict(X_test)
error_rf = mean_squared_error(y_test, predicciones_rf)

# Comparación
resultados = pd.DataFrame({
    "Modelo": ["Red Neuronal", "Random Forest"],
    "Error (MSE)": [error_nn, error_rf]
})

# Visualización
st.title("Comparación de Modelos")
st.table(resultados)

# Gráfico Red Neuronal
fig_nn = px.scatter(
    x=X_test.values.flatten(),
    y=y_test,
    labels={"x": "Año", "y": "Cantidad de Artículos"},
    title="Red Neuronal - Predicciones vs Reales"
)
fig_nn.add_scatter(x=X_test.values.flatten(), y=predicciones_nn.flatten(), mode='lines', name='Predicciones')
st.plotly_chart(fig_nn)

# Gráfico Random Forest
fig_rf = px.scatter(
    x=X_test.values.flatten(),
    y=y_test,
    labels={"x": "Año", "y": "Cantidad de Artículos"},
    title="Random Forest - Predicciones vs Reales"
)
fig_rf.add_scatter(x=X_test.values.flatten(), y=predicciones_rf, mode='lines', name='Predicciones')
st.plotly_chart(fig_rf)
