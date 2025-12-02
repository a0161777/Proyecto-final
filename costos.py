import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configuración básica
st.set_page_config(page_title="Predicción de costos", layout="centered")

st.write("# Predicción de tu costo")
st.image("inversiones.jpg", caption="Predicción de lo que te va a costar una actividad.")

# 1. Cargar datos
datos = pd.read_csv("costos_pred.csv", encoding="latin-1")

# 2. Mapear variables categóricas texto -> número
tipo_mapping = {nombre: idx + 1 for idx, nombre in enumerate(sorted(datos["Tipo"].unique()))}
momento_mapping = {nombre: idx + 1 for idx, nombre in enumerate(sorted(datos["Momento"].unique()))}

datos["Tipo_cod"] = datos["Tipo"].map(tipo_mapping)
datos["Momento_cod"] = datos["Momento"].map(momento_mapping)

# 3. Features y target
X = datos[["Presupuesto", "Tiempo invertido", "Tipo_cod", "Momento_cod", "No. de personas"]]
y = datos["Costo"]

# 4. Modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613777
)

LR = LinearRegression()
LR.fit(X_train, y_train)

st.header("Descripción de la actividad")

# 5. Entradas del usuario (con rangos realistas)
def user_input_features():
    Presupuesto = st.number_input(
        "Presupuesto de la actividad:",
        min_value=30.0,
        max_value=6000.0,
        value=100.0,
        step=10.0,
    )

    Tiempo = st.number_input(
        "Tiempo invertido en la actividad (minutos/horas):",
        min_value=2.0,
        max_value=180.0,
        value=15.0,
        step=1.0,
    )

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
        max_value=7.0,
        value=1.0,
        step=1.0,
    )

    user_input_data = {
        "Presupuesto": Presupuesto,
        "Tiempo invertido": Tiempo,
        "Tipo_cod": tipo_mapping[Tipo_txt],
        "Momento_cod": momento_mapping[Momento_txt],
        "No. de personas": Personas,
    }

    return pd.DataFrame(user_input_data, index=[0])

# Obtener inputs
df = user_input_features()

# Predicción
prediccion_bruta = LR.predict(df)[0]

# Asegurar que no haya negativos
prediccion = max(0.0, prediccion_bruta)

st.subheader("Cálculo del costo")
st.write(f"El costo de la actividad será de: **{prediccion:.2f}**")

# Mostrar predicción cruda (opcional)
st.caption(f"Predicción del modelo sin ajustar: {prediccion_bruta:.2f}")
