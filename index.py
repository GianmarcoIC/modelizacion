import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración Supabase
SUPABASE_URL = "https://peioqwvlxrgujotcuazt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBlaW9xd3ZseHJndWpvdGN1YXp0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQwMzM0MDUsImV4cCI6MjAzOTYwOTQwNX0.fLmClBVIcVGr_iKYTw79kPJUb12Iem7beooWfesNiXE"

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para obtener datos de la tabla
@st.cache_data
def get_data_from_supabase(table_name):
    response = supabase.table(table_name).select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        st.warning(f"No se encontraron datos en la tabla {table_name}.")
        return pd.DataFrame()

# Obtener datos
table_name = "modeliza"
data = get_data_from_supabase(table_name)

# Mostrar los datos en Streamlit
st.title("Modelo Predictivo para Ventas")
if not data.empty:
    st.subheader("Datos Cargados de Supabase")
    st.dataframe(data)

    # Preprocesamiento de datos
    st.subheader("Preprocesamiento de Datos")
    if data.isnull().sum().sum() > 0:
        st.warning("Se encontraron valores faltantes. Serán eliminados para el análisis.")
        data = data.dropna()
    else:
        st.success("No se encontraron valores faltantes.")

    # Normalización
    scaler = StandardScaler()
    features = ["advertising", "discount", "season"]
    target = "sales"
    X = data[features]
    y = data[target]
    X_scaled = scaler.fit_transform(X)

    st.write("Datos Normalizados")
    st.dataframe(pd.DataFrame(X_scaled, columns=features))

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modelado
    st.subheader("Modelado Predictivo")
    model_choice = st.radio("Selecciona el modelo:", ["Regresión Lineal", "Árbol de Decisión"])

    if model_choice == "Regresión Lineal":
        model = LinearRegression()
    elif model_choice == "Árbol de Decisión":
        model = DecisionTreeRegressor(random_state=42)

    # Entrenar modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    st.subheader("Evaluación del Modelo")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MAE (Error Absoluto Medio):** {mae:.2f}")
    st.write(f"**RMSE (Raíz del Error Cuadrático Medio):** {rmse:.2f}")
    st.write(f"**R² (Coeficiente de Determinación):** {r2:.2f}")

    # Optimización del modelo
    st.subheader("Optimización del Modelo")
    if model_choice == "Árbol de Decisión":
        params = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), params, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write("Mejores Hiperparámetros:", grid_search.best_params_)
        y_optimized_pred = best_model.predict(X_test)
        optimized_r2 = r2_score(y_test, y_optimized_pred)
        st.write(f"**R² Optimizado:** {optimized_r2:.2f}")

    # Visualización de resultados
    st.subheader("Visualización de Resultados")
    results_df = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    st.write("Resultados de Predicción")
    st.dataframe(results_df)

else:
    st.error("No se pudo cargar la tabla desde Supabase. Verifica la conexión o el nombre de la tabla.")
