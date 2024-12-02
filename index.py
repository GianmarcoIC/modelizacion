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



st.title("Machine Learning I - IDL1")


# Modelo de clasificación: Árbol de Decisión
st.subheader("Entrenamiento y Evaluación del Modelo de Clasificación")

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train_cls, y_train_cls)
y_pred_cls_dt = clf_dt.predict(X_test_cls)

# Clasificación con Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train_cls, y_train_cls)
y_pred_cls_rf = clf_rf.predict(X_test_cls)

# Métricas para clasificación
st.write("**Reporte de Clasificación: Árbol de Decisión**")
st.text(classification_report(y_test_cls, y_pred_cls_dt))

st.write("**Reporte de Clasificación: Random Forest**")
st.text(classification_report(y_test_cls, y_pred_cls_rf))

# Matriz de confusión para Árbol de Decisión
conf_matrix_dt = confusion_matrix(y_test_cls, y_pred_cls_dt)
st.write("**Matriz de Confusión: Árbol de Decisión**")
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
st.pyplot()

# Matriz de confusión para Random Forest
conf_matrix_rf = confusion_matrix(y_test_cls, y_pred_cls_rf)
st.write("**Matriz de Confusión: Random Forest**")
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens')
st.pyplot()

# Curvas ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test_cls, clf_dt.predict_proba(X_test_cls)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test_cls, clf_rf.predict_proba(X_test_cls)[:, 1])
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_rf = auc(fpr_rf, tpr_rf)

st.write("**Curvas ROC**")
plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, color='blue', label=f"Árbol de Decisión (AUC = {roc_auc_dt:.2f})")
plt.plot(fpr_rf, tpr_rf, color='green', label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curvas ROC para Modelos de Clasificación")
plt.legend()
st.pyplot()

# Modelo de regresión polinómica
st.subheader("Regresión Polinómica")

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_regression, test_size=0.2, random_state=42)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)
y_pred_poly = model_poly.predict(X_test_poly)

# Métricas de regresión polinómica
mae_poly = mean_absolute_error(y_test_poly, y_pred_poly)
rmse_poly = mean_squared_error(y_test_poly, y_pred_poly, squared=False)
r2_poly = r2_score(y_test_poly, y_pred_poly)

st.write("**Métricas de Evaluación: Regresión Polinómica**")
st.write(f"MAE: {mae_poly:.2f}, RMSE: {rmse_poly:.2f}, R²: {r2_poly:.2f}")

# Gráfico de dispersión para comparación
st.write("**Gráfico de Dispersión: Regresión Polinómica**")
plt.figure(figsize=(10, 6))
plt.scatter(y_test_poly, y_pred_poly, alpha=0.7, color='purple')
plt.plot([y_test_poly.min(), y_test_poly.max()], [y_test_poly.min(), y_test_poly.max()], 'k--', lw=2, color='red')
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Gráfico de Dispersión para Regresión Polinómica")
st.pyplot()

