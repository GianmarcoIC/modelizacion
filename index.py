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
st.title("Modelo Predictivo - Red Neuronal")

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

# Función CRUD
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
st.sidebar.title("CRUD")
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

st.write(f"Datos actuales en la tabla {selected_table}:")
st.dataframe(data)

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
        st.write("Tabla de predicciones Red Neuronal:")
        st.dataframe(predicciones_df)

        # Gráfico combinado: Histórico, predicciones y tendencia
        historico_df = datos_modelo.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
        historico_df["Tipo"] = "Histórico"
        predicciones_df["Tipo"] = "Predicción"
        grafico_df = pd.concat([historico_df, predicciones_df.rename(columns={"Predicción": "Cantidad de Artículos"})])

        fig = px.bar(
            grafico_df,
            x="Año",
            y="Cantidad de Artículos",
            color="Tipo",
            title="Publicaciones Históricas, Predicciones y Tendencia",
            barmode="group"
        )
        fig.add_scatter(x=predicciones_df["Año"], y=predicciones_df["Predicción"], mode="lines+markers", name="Tendencia")
        st.plotly_chart(fig)

        # Visualización de red neuronal
        st.subheader("Red Neuronal - Arquitectura")
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

        # Error del modelo
        errores = mean_squared_error(y_test, modelo_nn.predict(X_test))
        st.write(f"Error cuadrático medio (MSE): {errores:.4f}")

        # Después del modelo predictivo con red neuronal...

# Modelo predictivo con árbol de decisiones
st.title("Modelo de Predicción - Árbol de Decisiones")

try:
    # Obtener datos actualizados en tiempo real
    data = get_table_data("articulo")
    data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
    datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

    # Verificación de datos
    if datos_modelo.empty:
        st.warning("No hay datos suficientes para construir el modelo.")
    else:
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        # Entrenamiento del modelo
        modelo_arbol = DecisionTreeRegressor(random_state=42)
        modelo_arbol.fit(X, y)
        predicciones_arbol = modelo_arbol.predict(X)

        # Tabla de predicciones del árbol
        arbol_df = pd.DataFrame({
            "Año": X.values.flatten(),
            "Cantidad Real": y,
            "Predicción (Árbol)": predicciones_arbol
        })
        st.write("Predicciones del Árbol de Decisiones:")
        st.dataframe(arbol_df)

        # Visualización del árbol de decisión
        st.subheader("Árbol de Decisión - Estructura")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(modelo_arbol, feature_names=["anio_publicacion"], filled=True, rounded=True, fontsize=10, ax=ax)
        st.pyplot(fig)

        # Exportar estructura textual del árbol
        arbol_texto = export_text(modelo_arbol, feature_names=["anio_publicacion"])
        st.text("Estructura del árbol de decisión:")
        st.text(arbol_texto)

        # Gráfico comparativo
        st.subheader("Gráfico Comparativo de Predicciones")
        plt.figure(figsize=(10, 6))
        plt.plot(X, y, label="Cantidad Real", marker="o", linestyle="--", color="blue")
        plt.plot(X, predicciones_arbol, label="Predicción (Árbol)", marker="x", linestyle="-", color="red")
        plt.xlabel("Año de Publicación")
        plt.ylabel("Cantidad de Artículos")
        plt.title("Comparación entre Datos Reales y Predicciones")
        plt.legend()
        plt.grid()
        st.pyplot(plt.gcf())

        # Métrica del modelo
        mse_arbol = mean_squared_error(y, predicciones_arbol)
        st.write(f"Error cuadrático medio (MSE) del Árbol de Decisión: {mse_arbol:.4f}")

except Exception as e:
    # Manejo de errores
    st.error(f"Error en el modelo: {e}")


        
    
