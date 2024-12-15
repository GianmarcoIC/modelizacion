import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

# Configuración de predicción
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

# CRUD básico
st.sidebar.title("CRUD")
selected_table = st.sidebar.selectbox("Selecciona una tabla", ["articulo", "estudiante", "institucion", "indizacion"])
crud_action = st.sidebar.radio("Acción CRUD", ["Crear", "Actualizar", "Eliminar"])

data = get_table_data(selected_table)
fields = list(data.columns) if not data.empty else []

def insert_row(table_name, fields):
    data = {field: st.sidebar.text_input(f"Ingresar {field}") for field in fields if field != "id"}
    if st.sidebar.button("Insertar"):
        try:
            response = supabase.table(table_name).insert([data]).execute()
            if response.error:
                st.error(f"Error al insertar datos: {response.error}")
            else:
                st.success("Datos insertados correctamente.")
        except Exception as e:
            st.error(f"Error al insertar datos: {e}")

def update_row(table_name, fields):
    record_id = st.sidebar.number_input("ID del registro a actualizar", min_value=1, step=1)
    data = {field: st.sidebar.text_input(f"Nuevo valor para {field}") for field in fields if field != "id"}
    if st.sidebar.button("Actualizar"):
        try:
            response = supabase.table(table_name).update(data).eq("id", record_id).execute()
            if response.error:
                st.error(f"Error al actualizar datos: {response.error}")
            else:
                st.success("Datos actualizados correctamente.")
        except Exception as e:
            st.error(f"Error al actualizar datos: {e}")

def delete_row(table_name):
    record_id = st.sidebar.number_input("ID del registro a eliminar", min_value=1, step=1)
    if st.sidebar.button("Eliminar"):
        try:
            response = supabase.table(table_name).delete().eq("id", record_id).execute()
            if response.error:
                st.error(f"Error al eliminar datos: {response.error}")
            else:
                st.success("Datos eliminados correctamente.")
        except Exception as e:
            st.error(f"Error al eliminar datos: {e}")

if crud_action == "Crear":
    insert_row(selected_table, fields)
elif crud_action == "Actualizar":
    update_row(selected_table, fields)
elif crud_action == "Eliminar":
    delete_row(selected_table)

st.write(f"Datos actuales en la tabla {selected_table}:")
st.dataframe(data)

# Modelo predictivo y visualización
data = get_table_data("articulo")
if not data.empty:
    try:
        # Preparación de datos históricos
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

        # Normalización de datos
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        # División y entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
        modelo_nn = Sequential([
            Dense(10, activation='relu', input_dim=1),
            Dense(10, activation='relu'),
            Dense(1, activation='linear')
        ])
        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=150, verbose=0)

        # Predicción
        años_prediccion = list(range(inicio_prediccion, fin_prediccion + 1))
        años_normalizados = (pd.DataFrame(años_prediccion) - X.min().values[0]) / (X.max().values[0] - X.min().values[0])
        predicciones = modelo_nn.predict(años_normalizados)
        predicciones_desnormalizadas = predicciones * (y.max() - y.min()) + y.min()

        # DataFrame para predicciones
        predicciones_df = pd.DataFrame({
            "Año": años_prediccion,
            "Cantidad Predicha": predicciones_desnormalizadas.flatten()
        })

        # Combinación de datos históricos y predicciones
        historico_df = datos_modelo.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
        historico_df["Tipo"] = "Histórico"
        predicciones_df["Tipo"] = "Predicción"
        grafico_df = pd.concat([historico_df, predicciones_df.rename(columns={"Cantidad Predicha": "Cantidad de Artículos"})])

        # Gráfico combinado
        fig = px.bar(
            grafico_df,
            x="Año",
            y="Cantidad de Artículos",
            color="Tipo",
            title="Predicción de Artículos Publicados por Año",
            labels={"Año": "Año", "Cantidad de Artículos": "Cantidad de Artículos", "Tipo": "Datos"},
            barmode="group"
        )
        fig.add_scatter(x=predicciones_df["Año"], y=predicciones_df["Cantidad Predicha"], mode='lines+markers', name="Tendencia Predicha")
        st.plotly_chart(fig)

        # Tabla de predicciones y análisis textual
        st.write("Tabla de predicciones:")
        st.dataframe(predicciones_df)
        promedio_pred = predicciones_df["Cantidad Predicha"].mean()
        st.write(f"**Análisis de predicciones:** Se espera un promedio de publicaciones anuales de aproximadamente {promedio_pred:.2f} artículos en los años predichos.")
        
    except Exception as e:
        st.error(f"Error en el modelo predictivo: {e}")
