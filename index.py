# Predicciones y visualización de resultados
data = get_table_data("articulo")
if not data.empty:
    try:
        # Preparar datos históricos
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')

        # Normalización de datos
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        # División de datos y entrenamiento
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

        # Desnormalización de predicciones
        predicciones_desnormalizadas = predicciones * (y.max() - y.min()) + y.min()
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

        # Mostrar tabla de predicciones
        st.write("Tabla de predicciones:")
        st.dataframe(predicciones_df)

        # Resumen textual
        promedio_pred = predicciones_df["Cantidad Predicha"].mean()
        st.write(f"**Análisis de predicciones:** Se espera un promedio de publicaciones anuales de aproximadamente {promedio_pred:.2f} artículos en los años predichos.")
        
    except Exception as e:
        st.error(f"Error en el modelo predictivo: {e}")
