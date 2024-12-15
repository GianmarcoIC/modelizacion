import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from graphviz import Digraph


# Configuración Supabase
SUPABASE_URL = "https://ixgmctnuldngzludgets.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4Z21jdG51bGRuZ3psdWRnZXRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4ODQ4NjMsImV4cCI6MjA0OTQ2MDg2M30.T5LUIZCZA45OxtjTV2X9Ib6htozrrRdaKIjwgK1dsmg"

# Crear cliente Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para obtener datos
def get_data():
    data = supabase.table("articulo").select("*").execute().data
    df = pd.DataFrame(data)
    df['anio_publicacion'] = pd.to_numeric(df['anio_publicacion'], errors="coerce")
    return df.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

# CRUD en Streamlit
def crud_interface(table_name):
    data = supabase.table(table_name).select("*").execute().data
    df = pd.DataFrame(data)
    st.sidebar.title("CRUD")
    action = st.sidebar.radio("Acción", ["Insertar", "Actualizar", "Eliminar"])
    if action == "Insertar":
        fields = {col: st.sidebar.text_input(f"Ingresar {col}") for col in df.columns if col != "id"}
        if st.sidebar.button("Guardar"):
            supabase.table(table_name).insert(fields).execute()
    elif action == "Actualizar":
        record_id = st.sidebar.number_input("ID a actualizar", min_value=1)
        fields = {col: st.sidebar.text_input(f"Nuevo valor {col}") for col in df.columns if col != "id"}
        if st.sidebar.button("Actualizar"):
            supabase.table(table_name).update(fields).eq("id", record_id).execute()
    elif action == "Eliminar":
        record_id = st.sidebar.number_input("ID a eliminar", min_value=1)
        if st.sidebar.button("Eliminar"):
            supabase.table(table_name).delete().eq("id", record_id).execute()

# Obtener datos
df = get_data()

# División de datos
X = df[['anio_publicacion']]
y = df['cantidad_articulos']
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

# Comparación de modelos
resultados = pd.DataFrame({
    "Modelo": ["Red Neuronal", "Random Forest"],
    "Error (MSE)": [error_nn, error_rf]
})

# Mostrar resultados
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

# Visualización Red Neuronal
st.subheader("Visualización de la Red Neuronal")
nn_graph = Digraph(format="png")
nn_graph.attr(rankdir="LR")

nn_graph.node("Input", "Año", shape="circle", style="filled", color="lightblue")
for i in range(1, 6):
    nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
    nn_graph.edge("Input", f"Hidden1_{i}")
for i in range(1, 6):
    nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i}", shape="circle", style="filled", color="lightgreen")
    for j in range(1, 6):
        nn_graph.edge(f"Hidden1_{j}", f"Hidden2_{i}")
nn_graph.node("Output", "Predicción", shape="circle", style="filled", color="orange")
for i in range(1, 6):
    nn_graph.edge(f"Hidden2_{i}", "Output")

st.graphviz_chart(nn_graph)

# CRUD y gráfica de predicciones
crud_interface("articulo")
st.title("Histórico y Predicciones")
historico_df = df.rename(columns={"anio_publicacion": "Año", "cantidad_articulos": "Cantidad de Artículos"})
historico_df["Tipo"] = "Histórico"

predicciones_df = pd.DataFrame({
    "Año": X_test.values.flatten(),
    "Cantidad de Artículos": predicciones_nn.flatten(),
    "Tipo": "Predicción"
})
grafico_df = pd.concat([historico_df, predicciones_df])

fig = px.bar(
    grafico_df,
    x="Año",
    y="Cantidad de Artículos",
    color="Tipo",
    barmode="group",
    title="Publicaciones Históricas y Predicciones"
)
st.plotly_chart(fig)
