import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from graphviz import Digraph
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo - Red Neuronal 2024")

# Crear cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Opción para predecir años
st.sidebar.title("Configuración de Predicción")
inicio_prediccion = st.sidebar.number_input("Año inicial de predicción", value=2024, step=1)
fin_prediccion = st.sidebar.number_input("Año final de predicción", value=2026, step=1)

# Función para obtener datos de una tabla
def get_table_data(table_name):
    response = supabase.table(table_name).select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        st.warning(f"La tabla {table_name} está vacía.")
        return pd.DataFrame()

# Modelo predictivo con red neuronal
data = get_table_data("articulo")
if not data.empty:
    try:
        # Preprocesamiento de datos
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_normalized = scaler_X.fit_transform(X)
        y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Modelo mejorado
        modelo_nn = Sequential([
            Dense(64, activation='relu', input_dim=1),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])

        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0)

        # Predicción
        años_prediccion = list(range(inicio_prediccion, fin_prediccion + 1))
        años_normalizados = scaler_X.transform(pd.DataFrame(años_prediccion))
        predicciones_normalizadas = modelo_nn.predict(años_normalizados)
        predicciones = scaler_y.inverse_transform(predicciones_normalizadas)

        predicciones_df = pd.DataFrame({
            "Año": años_prediccion,
            "Predicción": predicciones.flatten()
        })
        st.write("Tabla de predicciones:")
        st.dataframe(predicciones_df)

        # Gráfico combinado: Histórico y predicciones
        historico_df = datos_modelo.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
        historico_df["Tipo"] = "Histórico"
        predicciones_df["Tipo"] = "Predicción"
        grafico_df = pd.concat([historico_df, predicciones_df.rename(columns={"Predicción": "Cantidad de Artículos"})])

        fig = px.bar(
            grafico_df,
            x="Año",
            y="Cantidad de Artículos",
            color="Tipo",
            title="Publicaciones Históricas y Predicciones",
            barmode="group"
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error en el modelo: {e}")
