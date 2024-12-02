import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuración de Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data
def get_data_from_supabase(table_name):
    try:
        response = supabase_client.table(table_name).select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.warning(f"La tabla '{table_name}' está vacía o no existe.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"No se pudo obtener datos de Supabase: {e}")
        return pd.DataFrame()

# Interfaz de usuario
st.title("Análisis de Modelos Predictivos")
data = get_data_from_supabase("variable")  # Cambia "variable" por el nombre real de la tabla.

if not data.empty:
    st.subheader("Datos Importados de Supabase")
    st.dataframe(data)

    # Preprocesamiento de datos
    st.subheader("Preprocesamiento de Datos")
    if data.isnull().sum().sum() > 0:
        st.warning("Se encontraron valores faltantes. Serán eliminados.")
        data = data.dropna()
    else:
        st.success("No se encontraron valores faltantes.")

    # Escalado de datos
    features = ["advertising", "discount", "season"]
    target = "sales"
    X = data[features]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("Datos Escalados")
    st.dataframe(pd.DataFrame(X_scaled, columns=features))

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modelado
    st.subheader("Entrenamiento y Evaluación del Modelo")

    # Modelo 1: Regresión Lineal
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    # Validación cruzada para regresión lineal
    cv_scores_lr = cross_val_score(model_lr, X_scaled, y, cv=5, scoring="r2")

    # Modelo 2: Árbol de Decisión
    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)

    # Validación cruzada para árbol de decisión
    cv_scores_dt = cross_val_score(model_dt, X_scaled, y, cv=5, scoring="r2")

    # Evaluación del modelo
    metrics = {
        "Modelo": ["Regresión Lineal", "Árbol de Decisión"],
        "MAE": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_dt),
        ],
        "RMSE": [
            mean_squared_error(y_test, y_pred_lr, squared=False),
            mean_squared_error(y_test, y_pred_dt, squared=False),
        ],
        "R²": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_dt),
        ],
        "R² Validación Cruzada": [
            np.mean(cv_scores_lr),
            np.mean(cv_scores_dt),
        ],
    }

    st.write("**Métricas de Evaluación**")
    st.dataframe(pd.DataFrame(metrics))

    # Comparación de resultados
    st.subheader("Comparación de Resultados")
    results_df = pd.DataFrame({
        "Real": y_test,
        "Predicción (Regresión Lineal)": y_pred_lr,
        "Predicción (Árbol de Decisión)": y_pred_dt,
    })
    st.dataframe(results_df)

    # Descarga de informe
    st.download_button(
        "Descargar Informe",
        data=results_df.to_csv(index=False),
        file_name="reporte_modelo.csv",
        mime="text/csv",
    )
else:
    st.error("No se encontraron datos en la tabla o la conexión falló.")
