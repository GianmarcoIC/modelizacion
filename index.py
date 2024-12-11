import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
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








st.title("Machine Learning I - IDL1")

# División de datos (Regresión: variable objetivo 'sales')
X_regression = data[["advertising", "discount", "season"]]
y_regression = data["sales"]

# Escalar las características para regresión
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_regression)

# División de datos para entrenamiento y prueba (Regresión)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_regression, test_size=0.2, random_state=42
)

# Modelo de Regresión Polinómica
st.subheader("Regresión Polinómica")
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_reg)
y_pred_poly = model_poly.predict(X_test_poly)

# Métricas para Regresión Polinómica
mae_poly = mean_absolute_error(y_test_reg, y_pred_poly)
rmse_poly = mean_squared_error(y_test_reg, y_pred_poly, squared=False)
r2_poly = r2_score(y_test_reg, y_pred_poly)

st.write("**Métricas de Evaluación: Regresión Polinómica**")
st.write(f"MAE: {mae_poly:.2f}, RMSE: {rmse_poly:.2f}, R²: {r2_poly:.2f}")

# Gráfico de dispersión para Regresión Polinómica
st.write("**Gráfico de Dispersión: Regresión Polinómica**")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_reg, y_pred_poly, alpha=0.7, color='purple')
ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2, color='red')
ax.set_xlabel("Valores Reales")
ax.set_ylabel("Predicciones")
ax.set_title("Gráfico de Dispersión para Regresión Polinómica")
st.pyplot(fig)

# División de datos (Clasificación: variable objetivo 'season')
X_classification = data[["advertising", "discount", "sales"]]
y_classification = data["season"]

# Escalar las características para clasificación
X_cls_scaled = scaler.fit_transform(X_classification)

# División de datos para entrenamiento y prueba (Clasificación)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls_scaled, y_classification, test_size=0.2, random_state=42
)

# Modelo Árbol de Decisión
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train_cls, y_train_cls)
y_pred_cls_dt = clf_dt.predict(X_test_cls)

st.subheader("Clasificación con Árbol de Decisión")
st.write("**Reporte de Clasificación: Árbol de Decisión**")
st.text(classification_report(y_test_cls, y_pred_cls_dt))

# Matriz de confusión para Árbol de Decisión
conf_matrix_dt = confusion_matrix(y_test_cls, y_pred_cls_dt)
st.write("**Matriz de Confusión: Árbol de Decisión**")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Modelo Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train_cls, y_train_cls)
y_pred_cls_rf = clf_rf.predict(X_test_cls)

st.subheader("Clasificación con Random Forest")
st.write("**Reporte de Clasificación: Random Forest**")
st.text(classification_report(y_test_cls, y_pred_cls_rf))

# Matriz de confusión para Random Forest
conf_matrix_rf = confusion_matrix(y_test_cls, y_pred_cls_rf)
st.write("**Matriz de Confusión: Random Forest**")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', ax=ax)
st.pyplot(fig)






import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests

# Título de la aplicación
st.title("Análisis Computacional de Datos")

# Extracción de datos con BeautifulSoup
st.subheader("Extracción de Datos")
def extract_data_from_web(url):
    """
    Función para extraer datos de una página web usando BeautifulSoup.
    Valida si la página contiene una tabla antes de procesar.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Validar si hay tablas en la página
        table = soup.find('table')
        if table:
            st.success("Tabla encontrada en la página web.")
            df = pd.read_html(str(table))[0]
            return df
        else:
            st.error("No se encontraron tablas en la página web proporcionada.")
            return None
    except Exception as e:
        st.error(f"Error al procesar la URL: {e}")
        return None

# URL de ejemplo con validador
url = st.text_input("Ingrese una URL con una tabla de datos:", "https://example.com/data")
data_web = extract_data_from_web(url)

if data_web is not None:
    st.write("Datos extraídos de la web:")
    st.dataframe(data_web)
else:
    st.warning("No se pudieron extraer datos desde la URL proporcionada.")

# Cargar archivo CSV adicional
st.subheader("Carga de Datos desde un Archivo CSV")
uploaded_file = st.file_uploader("Sube un archivo CSV:", type="csv")
data_csv = pd.read_csv(uploaded_file) if uploaded_file else None

if data_csv is not None:
    st.write("Datos cargados desde el archivo CSV:")
    st.dataframe(data_csv)

# Limpieza de datos
st.subheader("Limpieza de Datos")
def clean_data(df):
    """
    Función para limpiar datos: manejar valores nulos, duplicados y outliers.
    """
    if df is not None:
        st.write(f"**Datos antes de limpieza (filas, columnas): {df.shape}**")
        # Manejo de valores nulos
        df = df.dropna()
        st.write(f"**Filas restantes después de eliminar valores nulos: {df.shape[0]}**")
        
        # Eliminar duplicados
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        st.write(f"**Duplicados eliminados: {initial_rows - df.shape[0]}**")
        
        # Manejo de outliers (IQR)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        st.write(f"**Outliers detectados: {outliers.sum()}**")
        df = df[~outliers]
        st.write(f"**Datos después de limpieza (filas, columnas): {df.shape}**")
        return df
    else:
        st.warning("No hay datos para limpiar.")
        return None

data_web_cleaned = clean_data(data_web)
data_csv_cleaned = clean_data(data_csv)

# Integración de datos
st.subheader("Integración de Datos")
def integrate_data(df1, df2, key=None):
    """
    Función para integrar datos de diferentes fuentes.
    """
    if df1 is not None and df2 is not None:
        if key:
            integrated_df = pd.merge(df1, df2, on=key)
            st.write(f"**Datos integrados utilizando clave '{key}': {integrated_df.shape}**")
        else:
            integrated_df = pd.concat([df1, df2], ignore_index=True)
            st.write(f"**Datos integrados (filas, columnas): {integrated_df.shape}**")
        return integrated_df
    else:
        st.warning("No se pueden integrar los datos: faltan fuentes.")
        return None

final_dataset = integrate_data(data_web_cleaned, data_csv_cleaned)

# Documentación del proceso
st.subheader("Documentación del Proceso")
def generate_report(df, step_name):
    """
    Genera un informe detallado del estado de los datos en cada paso.
    """
    if df is not None:
        report = {
            'Paso': step_name,
            'Filas y Columnas': df.shape,
            'Valores Nulos': df.isnull().sum().to_dict(),
            'Duplicados': df.duplicated().sum(),
        }
    else:
        report = {'Paso': step_name, 'Estado': 'Sin datos'}
    return report

# Informes
report_extraction = generate_report(data_web, "Extracción de Datos")
report_cleaning_web = generate_report(data_web_cleaned, "Limpieza de Datos Web")
report_cleaning_csv = generate_report(data_csv_cleaned, "Limpieza de Datos CSV")
report_integration = generate_report(final_dataset, "Integración de Datos")

# Mostrar informes
st.write("**Informe de Extracción:**", report_extraction)
st.write("**Informe de Limpieza de Datos Web:**", report_cleaning_web)
st.write("**Informe de Limpieza de Datos CSV:**", report_cleaning_csv)
st.write("**Informe de Integración:**", report_integration)

# Guardar el conjunto de datos final
if final_dataset is not None:
    st.subheader("Descarga del Conjunto de Datos Final")
    csv_data = final_dataset.to_csv(index=False)
    st.download_button(
        label="Descargar Dataset Final",
        data=csv_data,
        file_name="final_dataset.csv",
        mime="text/csv"
    )







