import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from supabase import create_client

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logo
st.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Logo_de_Streamlit.png", width=150)

# Función para cargar datos de Supabase
def load_data(table_name):
    data = supabase.table(table_name).select("*").execute().data
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

# Función para guardar datos en Supabase
def save_data(table_name, record):
    supabase.table(table_name).insert(record).execute()

# Función para actualizar datos en Supabase
def update_data(table_name, record_id, record):
    supabase.table(table_name).update(record).eq("id", record_id).execute()

# Función para eliminar datos en Supabase
def delete_data(table_name, record_id):
    supabase.table(table_name).delete().eq("id", record_id).execute()

# CRUD por tabla
def crud_table(table_name):
    st.subheader(f"Gestión de {table_name}")
    data = load_data(table_name)

    if data.empty:
        st.warning(f"No hay datos disponibles en la tabla {table_name}.")
        return

    st.dataframe(data)

    with st.expander("Insertar nuevo registro"):
        fields = {col: st.text_input(f"{col}") for col in data.columns if col != "id"}
        if st.button("Guardar", key=f"insert_{table_name}"):
            save_data(table_name, fields)
            st.success("Registro guardado exitosamente")

    with st.expander("Actualizar registro existente"):
        record_id = st.number_input("ID del registro", min_value=1, step=1, key=f"update_id_{table_name}")
        fields = {col: st.text_input(f"Nuevo {col}") for col in data.columns if col != "id"}
        if st.button("Actualizar", key=f"update_{table_name}"):
            update_data(table_name, record_id, fields)
            st.success("Registro actualizado exitosamente")

    with st.expander("Eliminar registro"):
        record_id = st.number_input("ID del registro", min_value=1, step=1, key=f"delete_id_{table_name}")
        if st.button("Eliminar", key=f"delete_{table_name}"):
            delete_data(table_name, record_id)
            st.success("Registro eliminado exitosamente")

# Modelos predictivos
def prediction_model():
    df = load_data("Articulo")

    if df.empty or "anio_publicacion" not in df.columns:
        st.warning("No hay datos suficientes para realizar la predicción.")
        return

    df["anio_publicacion"] = pd.to_numeric(df["anio_publicacion"], errors="coerce")
    df = df.dropna(subset=["anio_publicacion"])

    data = df.groupby("anio_publicacion").size().reset_index(name="cantidad")
    X = data[["anio_publicacion"]]
    y = data["cantidad"]

    if len(X) < 2:
        st.warning("No hay suficientes datos históricos para entrenar el modelo.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    st.write(f"Error Cuadrático Medio (MSE): {mse}")

    # Tendencia de predicción
    future_years = pd.DataFrame({"anio_publicacion": range(data["anio_publicacion"].max() + 1, data["anio_publicacion"].max() + 6)})
    future_predictions = model.predict(future_years)

    combined_data = pd.concat([
        data.rename(columns={"cantidad": "valor"}).assign(tipo="Histórico"),
        future_years.assign(valor=future_predictions, tipo="Predicción")
    ])

    # Gráficos
    st.subheader("Tendencia de Publicaciones")
    fig = px.bar(combined_data, x="anio_publicacion", y="valor", color="tipo", barmode="group", title="Tendencia de Publicaciones")
    st.plotly_chart(fig)

# CRUD para todas las tablas
for table in ["Estudiante", "Institucion", "Indizacion", "Articulo"]:
    crud_table(table)

# Modelo de predicción
prediction_model()
