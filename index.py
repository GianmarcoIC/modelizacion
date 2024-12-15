import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from graphviz import Digraph
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo - Comparación de Métodos 2024")

# Crear cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Opción para predecir años
st.sidebar.title("Configuración de Predicción")
inicio_prediccion = st.sidebar.number_input("Año inicial de predicción", value=2024, step=1)
fin_prediccion = st.sidebar.number_input("Año final de predicción", value=2026, step=1)

# Función para obtener datos de una tabla
def get_table_data(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.warning(f"La tabla {table_name} está vacía.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al consultar la tabla {table_name}: {e}")
        return pd.DataFrame()

# Obtener datos
data = get_table_data("articulo")
if not data.empty:
    try:
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        # Normalización
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Modelo Red Neuronal
        modelo_nn = Sequential([
            Dense(5, activation='relu', input_dim=1),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')
        ])
        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)

        # Modelo Árbol de Decisión
        modelo_tree = DecisionTreeRegressor(random_state=42)
        modelo_tree.fit(X_train, y_train)

        # Modelo Random Forest
        modelo_rf = RandomForestRegressor(random_state=42, n_estimators=100)
        modelo_rf.fit(X_train, y_train)

        # Predicción
        años_prediccion = list(range(inicio_prediccion, fin_prediccion + 1))
        años_normalizados = (pd.DataFrame(años_prediccion) - X.min().values[0]) / (X.max().values[0] - X.min().values[0])

        pred_nn = modelo_nn.predict(años_normalizados)
        pred_tree = modelo_tree.predict(años_normalizados)
        pred_rf = modelo_rf.predict(años_normalizados)

        # Desnormalizar
        pred_nn_denorm = pred_nn * (y.max() - y.min()) + y.min()
        pred_tree_denorm = pred_tree * (y.max() - y.min()) + y.min()
        pred_rf_denorm = pred_rf * (y.max() - y.min()) + y.min()

        # Crear DataFrame de predicciones
        predicciones_df = pd.DataFrame({
            "Año": años_prediccion,
            "Red Neuronal": pred_nn_denorm.flatten(),
            "Árbol de Decisión": pred_tree_denorm,
            "Random Forest": pred_rf_denorm
        })

        st.write("Tabla comparativa de predicciones:")
        st.dataframe(predicciones_df)

        # Gráfico combinado
        historico_df = datos_modelo.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
        historico_df["Tipo"] = "Histórico"
        predicciones_long = predicciones_df.melt(id_vars="Año", var_name="Modelo", value_name="Cantidad de Artículos")

        grafico_df = pd.concat([historico_df, predicciones_long.rename(columns={"Cantidad de Artículos": "Cantidad de Artículos"})])

        fig = px.bar(
            grafico_df,
            x="Año",
            y="Cantidad de Artículos",
            color="Modelo",
            title="Comparación de Predicciones",
            barmode="group"
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error en el modelo: {e}")
