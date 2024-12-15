import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from graphviz import Digraph
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo - Red Neuronal 2024")

# Crear cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

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

# Funciones CRUD
def insert_row(table_name, fields):
    data = {field: st.sidebar.text_input(f"Ingresar {field}") for field in fields if field != "id"}
    if st.sidebar.button("Insertar"):
        try:
            insert_data(table_name, [data])
        except Exception as e:
            st.error(f"Error al insertar datos: {e}")

def update_row(table_name, fields):
    record_id = st.sidebar.number_input("ID del registro a actualizar", min_value=1, step=1)
    data = {field: st.sidebar.text_input(f"Nuevo valor para {field}") for field in fields if field != "id"}
    if st.sidebar.button("Actualizar"):
        try:
            update_data(table_name, record_id, {k: v for k, v in data.items() if v})
        except Exception as e:
            st.error(f"Error al actualizar datos: {e}")

def delete_row(table_name):
    record_id = st.sidebar.number_input("ID del registro a eliminar", min_value=1, step=1)
    if st.sidebar.button("Eliminar"):
        delete_data(table_name, record_id)

def insert_data(table_name, data):
    response = supabase.table(table_name).insert(data).execute()
    if response.error:
        st.error(f"Error al insertar datos en {table_name}: {response.error}")
    else:
        st.success(f"Datos insertados correctamente en {table_name}.")

def update_data(table_name, id_value, data):
    response = supabase.table(table_name).update(data).eq("id", id_value).execute()
    if response.error:
        st.error(f"Error al actualizar datos en {table_name}: {response.error}")
    else:
        st.success(f"Datos actualizados correctamente en {table_name}.")

def delete_data(table_name, id_value):
    response = supabase.table(table_name).delete().eq("id", id_value).execute()
    if response.error:
        st.error(f"Error al eliminar datos en {table_name}: {response.error}")
    else:
        st.success(f"Datos eliminados correctamente.")

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
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        # Normalización
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Modelo
        modelo_nn = Sequential([
            Dense(5, activation='relu', input_dim=1),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')
        ])
        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)

        # Predicción
        años_prediccion = list(range(inicio_prediccion, fin_prediccion + 1))
        años_normalizados = (pd.DataFrame(años_prediccion) - X.min().values[0]) / (X.max().values[0] - X.min().values[0])
        predicciones = modelo_nn.predict(años_normalizados)

        predicciones_desnormalizadas = predicciones * (y.max() - y.min()) + y.min()
        predicciones_df = pd.DataFrame({
            "Año": años_prediccion,
            "Predicción": predicciones_desnormalizadas.flatten()
        })
        st.write("Tabla de predicciones:")
        st.dataframe(predicciones_df)

        # Visualización de red neuronal
        st.subheader("Red Neuronal con Valores")
        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        nn_graph.node("Input", f"Año [{X.mean().values[0]:.2f}]", shape="circle", style="filled", color="lightblue")
        for i in range(1, 6):
            nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i} [{modelo_nn.get_weights()[0][0][i-1]:.2f}]", shape="circle", style="filled", color="lightgreen")
        for i in range(1, 6):
            nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i} [{modelo_nn.get_weights()[2][0][i-1]:.2f}]", shape="circle", style="filled", color="lightgreen")
        nn_graph.node("Output", f"Predicción [{predicciones_df['Predicción'].mean():.2f}]", shape="circle", style="filled", color="orange")

        nn_graph.edge("Input", "Hidden1_1")
        for i in range(1, 6):
            for j in range(1, 6):
                nn_graph.edge(f"Hidden1_{i}", f"Hidden2_{j}")
        for i in range(1, 6):
            nn_graph.edge(f"Hidden2_{i}", "Output")

        st.graphviz_chart(nn_graph)

    except Exception as e:
        st.error(f"Error en el modelo: {e}")
# Gráfico combinado: Histórico y predicciones
data = get_table_data("articulo")
if not data.empty:
    try:
        # Agrupar datos históricos
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

        # Crear DataFrame de predicciones si existen
        if 'predicciones_df' in locals():
            historico_df = datos_modelo.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
            historico_df["Tipo"] = "Histórico"
            predicciones_df["Tipo"] = "Predicción"
            grafico_df = pd.concat([historico_df, predicciones_df.rename(columns={"Predicción": "Cantidad de Artículos"})])

            # Crear el gráfico de barras
            fig = px.bar(
                grafico_df,
                x="Año",
                y="Cantidad de Artículos",
                color="Tipo",
                title="Publicaciones Históricas y Predicciones",
                labels={"Año": "Año", "Cantidad de Artículos": "Cantidad de Artículos", "Tipo": "Datos"},
                barmode="group"
            )
            st.plotly_chart(fig)
        else:
            st.warning("No se encontraron predicciones para visualizar.")

    except Exception as e:
        st.error(f"Error al generar el gráfico de barras: {e}")



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
