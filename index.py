import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text
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

        # Visualización de red neuronal con valores
        st.subheader("Red Neuronal - Arquitectura y Predicciones")
        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        nn_graph.node("Input", "Año", shape="circle", style="filled", color="lightblue")
        for i in range(1, 65):
            nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
        for i in range(1, 33):
            nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i}", shape="circle", style="filled", color="lightyellow")
        nn_graph.node("Output", "Predicción\n(Valores)", shape="circle", style="filled", color="orange")

        nn_graph.edge("Input", "Hidden1_1")
        for i in range(1, 65):
            for j in range(1, 33):
                nn_graph.edge(f"Hidden1_{i}", f"Hidden2_{j}")
        for i in range(1, 33):
            nn_graph.edge(f"Hidden2_{i}", "Output")

        nn_graph.node("Output", f"Predicción\n{', '.join([str(round(val, 2)) for val in predicciones.flatten()])}", shape="circle", style="filled", color="orange")

        st.graphviz_chart(nn_graph)

        # Error del modelo
        errores = mean_squared_error(y_test, modelo_nn.predict(X_test))
        st.write(f"Error cuadrático medio (MSE): {errores:.4f}")

        # Modelo predictivo con árbol de decisiones
        st.title("Modelo de Predicción - Árbol de Decisiones")
        modelo_arbol = DecisionTreeRegressor(random_state=42)
        modelo_arbol.fit(X, y)
        predicciones_arbol = modelo_arbol.predict(X)

        # Tabla de predicciones del árbol
        arbol_df = pd.DataFrame({
            "Año": X.flatten(),
            "Predicción (Árbol)": predicciones_arbol
        })
        st.write("Predicciones del Árbol de Decisiones:")
        st.dataframe(arbol_df)

        # Tabla comparativa de modelos
        comparacion_df = pd.DataFrame({
            "Año": años_prediccion,
            "Predicción NN": predicciones.flatten(),
            "Predicción Árbol": modelo_arbol.predict(pd.DataFrame(años_prediccion))
        })
        st.write("Comparación de modelos predictivos:")
        st.dataframe(comparacion_df)

    except Exception as e:
        st.error(f"Error en el modelo: {e}")
