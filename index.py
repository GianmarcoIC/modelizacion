import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from supabase import create_client, Client

# Configuración de Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

# Crear cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Imagen y título
st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo y Gestión de Datos")

# Función para obtener datos de una tabla
def get_table_data(table_name):
    response = supabase.table(table_name).select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        st.warning(f"La tabla {table_name} está vacía.")
        return pd.DataFrame()

# Funciones CRUD
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

# CRUD en la barra lateral
st.sidebar.title("Gestión de Datos (CRUD)")
selected_table = st.sidebar.selectbox("Selecciona una tabla", ["articulo", "estudiante", "institucion", "indizacion"])
crud_action = st.sidebar.radio("Acción CRUD", ["Crear", "Actualizar", "Eliminar"])

data = get_table_data(selected_table)
fields = list(data.columns) if not data.empty else []

if crud_action == "Crear":
    insert_row(selected_table, fields)
elif crud_action == "Actualizar":
    update_row(selected_table, fields)
elif crud_action == "Eliminar":
    delete_row(selected_table)

# Mostrar datos actuales
st.write(f"Datos actuales en la tabla {selected_table}:")
st.dataframe(data)

# Modelo de predicción - Árbol de Decisión
st.title("Modelo Predictivo - Árbol de Decisión")

try:
    # Preprocesamiento de datos
    data = get_table_data("articulo")
    data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
    datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
    X = datos_modelo[['anio_publicacion']]
    y = datos_modelo['cantidad_articulos']

    # Escalado de datos
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # División de los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de árbol de decisión
    modelo_arbol = DecisionTreeRegressor(random_state=42)
    modelo_arbol.fit(X_train, y_train)

    # Predicción
    y_pred_arbol = modelo_arbol.predict(X_test)

    # Visualización del árbol
    st.subheader("Árbol de Decisión - Estructura")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(modelo_arbol, feature_names=['anio_publicacion'], filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)

    # Exportar estructura del árbol como texto
    arbol_texto = export_text(modelo_arbol, feature_names=['anio_publicacion'])
    st.text("Estructura del árbol de decisión:")
    st.text(arbol_texto)

    # Error del modelo
    mse_arbol = mean_squared_error(y_test, y_pred_arbol)
    st.write(f"Error cuadrático medio (MSE) del Árbol de Decisión: {mse_arbol:.4f}")

except Exception as e:
    st.error(f"Error en el modelo: {e}")
