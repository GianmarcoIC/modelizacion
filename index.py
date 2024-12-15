import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from graphviz import Digraph
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo - Red Neuronal y Random Forest")

# Crear cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Opciones para predicción
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

# CRUD
st.sidebar.title("CRUD")
selected_table = st.sidebar.selectbox("Selecciona una tabla", ["articulo", "estudiante", "institucion", "indizacion"])
crud_action = st.sidebar.radio("Acción CRUD", ["Crear", "Actualizar", "Eliminar"])

data = get_table_data(selected_table)
fields = list(data.columns) if not data.empty else []

def insert_row(table_name, fields):
    data = {field: st.sidebar.text_input(f"Ingresar {field}") for field in fields if field != "id"}
    if st.sidebar.button("Insertar"):
        try:
            supabase.table(table_name).insert([data]).execute()
            st.success("Registro insertado correctamente")
        except Exception as e:
            st.error(f"Error al insertar datos: {e}")

def update_row(table_name, fields):
    record_id = st.sidebar.number_input("ID del registro a actualizar", min_value=1, step=1)
    data = {field: st.sidebar.text_input(f"Nuevo valor para {field}") for field in fields if field != "id"}
    if st.sidebar.button("Actualizar"):
        try:
            supabase.table(table_name).update(data).eq("id", record_id).execute()
            st.success("Registro actualizado correctamente")
        except Exception as e:
            st.error(f"Error al actualizar datos: {e}")

def delete_row(table_name):
    record_id = st.sidebar.number_input("ID del registro a eliminar", min_value=1, step=1)
    if st.sidebar.button("Eliminar"):
        try:
            supabase.table(table_name).delete().eq("id", record_id).execute()
            st.success("Registro eliminado correctamente")
        except Exception as e:
            st.error(f"Error al eliminar datos: {e}")

if crud_action == "Crear":
    insert_row(selected_table, fields)
elif crud_action == "Actualizar":
    update_row(selected_table, fields)
elif crud_action == "Eliminar":
    delete_row(selected_table, fields)

st.write(f"Datos actuales en la tabla {selected_table}:")
st.dataframe(data)

# Red Neuronal
st.header("Predicción con Red Neuronal")
data = get_table_data("articulo")
if not data.empty:
    try:
        # Preprocesamiento de datos para Red Neuronal
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_normalized = scaler_X.fit_transform(X)
        y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Modelo Red Neuronal
        modelo_nn = Sequential([
            Dense(64, activation='relu', input_dim=1),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])

        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0)

        # Predicción con Red Neuronal
        años_prediccion = list(range(inicio_prediccion, fin_prediccion + 1))
        años_normalizados = scaler_X.transform(pd.DataFrame(años_prediccion))
        predicciones_nn_normalizadas = modelo_nn.predict(años_normalizados)
        predicciones_nn = scaler_y.inverse_transform(predicciones_nn_normalizadas)

        predicciones_nn_df = pd.DataFrame({
            "Año": años_prediccion,
            "Predicción_NN": predicciones_nn.flatten()
        })

        st.write("Tabla de predicciones Red Neuronal:")
        st.dataframe(predicciones_nn_df)

        # Gráfico de predicciones Red Neuronal
        fig_nn = px.line(predicciones_nn_df, x="Año", y="Predicción_NN",
                         title="Predicciones Red Neuronal",
                         labels={"Predicción_NN": "Cantidad de Artículos"})
        st.plotly_chart(fig_nn)

        # Error del modelo Red Neuronal
        errores_nn = mean_squared_error(y_test, modelo_nn.predict(X_test))
        st.write(f"Error cuadrático medio Red Neuronal (MSE): {errores_nn:.4f}")

        # Visualización de Red Neuronal
        st.subheader("Arquitectura de la Red Neuronal")
        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        nn_graph.node("Input", "Año", shape="circle", style="filled", color="lightblue")
        for i in range(1, 65):
            nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
        for i in range(1, 33):
            nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i}", shape="circle", style="filled", color="lightyellow")
        nn_graph.node("Output", "Predicción", shape="circle", style="filled", color="orange")

        nn_graph.edge("Input", "Hidden1_1")
        for i in range(1, 65):
            for j in range(1, 33):
                nn_graph.edge(f"Hidden1_{i}", f"Hidden2_{j}")
        for i in range(1, 33):
            nn_graph.edge(f"Hidden2_{i}", "Output")

        st.graphviz_chart(nn_graph)

    except Exception as e:
        st.error(f"Error en el modelo Red Neuronal: {e}")

# Random Forest
st.header("Predicción con Random Forest")
if not data.empty:
    try:
        # Modelo Random Forest
        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_rf.fit(X_train, y_train.ravel())

        # Predicción con Random Forest
        predicciones_rf_normalizadas = modelo_rf.predict(años_normalizados)
        predicciones_rf = scaler_y.inverse_transform(predicciones_rf_normalizadas.reshape(-1, 1))

        predicciones_rf_df = pd.DataFrame({
            "Año": años_prediccion,
            "Predicción_RF": predicciones_rf.flatten()
        })

        st.write("Tabla de predicciones Random Forest:")
        st.dataframe(predicciones_rf_df)

        # Gráfico de predicciones Random Forest
        fig_rf = px.line(predicciones_rf_df, x="Año", y="Predicción_RF",
                         title="Predicciones Random Forest",
                         labels={"Predicción_RF": "Cantidad de Artículos"})
        st.plotly_chart(fig_rf)

        # Error del modelo Random Forest
        errores_rf = mean_squared_error(y_test, modelo_rf.predict(X_test))
        st.write(f"Error cuadrático medio Random Forest (MSE): {errores_rf:.4f}")

    except Exception as e:
        st.error(f"Error en el modelo Random Forest: {e}")
