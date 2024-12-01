import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración de Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

# Crear cliente de Supabase
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para obtener datos de Supabase
@st.cache_data
def get_data_from_supabase(table_name):
    """
    Obtiene datos de una tabla en Supabase.
    """
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
st.title("Análisis de Modelos Predictivo")
data = get_data_from_supabase("modeliza")  # Cambia "modeliza" por el nombre real de la tabla si es necesario

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

    # Escalar datos
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
    st.subheader("Entrenamiento del Modelo")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    st.subheader("Evaluación del Modelo")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    # Resultados de predicción
    st.subheader("Comparación de Resultados")
    results_df = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    st.dataframe(results_df)

else:
    st.error("No se encontraron datos en la tabla o la conexión falló.")
