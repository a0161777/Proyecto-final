import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("# Predicción de tu costo")
st.image("inversiones.jpg", caption="Predicción de lo que te va a costar una actividad.")

# 1. Cargar datos
datos = pd.read_csv("costos_pred.csv")  # utf-8 por defecto

# 2. Codificar variables categóricas (Texto -> número)
# Creamos diccionarios de mapeo a partir de los valores del CSV
tipo_mapping = {nombre: idx + 1 for idx, nombre in enumerate(sorted(datos["Tipo"].unique()))}
momento_mapping = {nombre: idx + 1 for idx, nombre in enumerate(sorted(datos["Momento"].unique()))}

# Agregamos columnas numéricas
datos["Tipo_cod"] = datos["Tipo"].map(tipo_mapping)
datos["Momento_cod"] = datos["Momento"].map(momento_mapping)

# 3. Definir X (features) e y (target)
X = datos[["Presupuesto", "Tiempo invertido", "Tipo_cod", "Momento_cod", "No. de personas"]]
y = datos["Costo"]

# 4. Separar en entrenamiento y prueba y entrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613797
)

LR = LinearRegression()
LR.fit(X_train, y_train)

st.header("Descripción de la actividad")

# 5. Entradas del usuario
def user_input_features():
    Presupuesto = st.number_input(
        "Presupuesto de la actividad:",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
    )
    Tiempo = st.number_input(
        "Tiempo invertido en la actividad (minutos/horas):",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
    )

    # Usamos las mismas categorías que vienen del CSV
    Tipo_txt = st.selectbox(
        "Tipo de actividad:",
        options=list(tipo_mapping.keys()),
    )

    Momento_txt = st.selectbox(
        "Momento del día en el que tu actividad se desarrolla:",
        options=list(momento_mapping.keys()),
    )

    Personas = st.number_input(
        "Número de personas involucradas en el gasto:",
        min_value=1.0,
        max_value=50.0,
        value=1.0,
    )

    user_input_data = {
        "Presupuesto": Presupuesto,
        "Tiempo invertido": Tiempo,
        "Tipo_cod": tipo_mapping[Tipo_txt],
        "Momento_cod": momento_mapping[Momento_txt],
        "No. de personas": Personas,
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

# 6. Obtener datos del usuario y predecir
df = user_input_features()

prediccion = LR.predict(df)[0]

st.subheader("Cálculo del costo")
st.write(f"El costo de la actividad será de: **{prediccion:.2f}**")
