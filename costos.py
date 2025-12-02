import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de tu costo ''')
st.image("inversiones.jpg", caption="Predicción de lo que te va a costar una actividad.")

st.header('Descripción de la actividad')

def user_input_features():
  # Entradas del usuario
  Presupuesto = st.number_input('Presupuesto de la actividad:', min_value=0, max_value=10000, value = 0)
  Tiempo = st.number_input('Tiempo invertido en la actividad',  min_value=0, max_value=10000, value = 0)
  Tipo = st.number_input('Tipo de actividad 1=Alimentos/salud, 2=Servicios, 3=Ejercicio/Deporte, 4=Ocio, 5=Académico, 6=Transporte', min_value=1, max_value=6, value = 1, step = 1)
  Momento = st.number_input('Momento del día en el que tu actividad se desarrolla, 1=Mañana, 2=Tarde, 3=Noche:', min_value=1, max_value=3, value = 1, step = 1)
  Personas = st.number_input('Número de personas invlolucradas en el gasto', min_value=1, max_value=50, value=1, step = 1)

#Utilizamos los nombres de nuestro conjunto de datos
  user_input_data = {'Presupuesto': Presupuesto,
                     'Tiempo invertido': Tiempo,
                     'Tipo': Tipo,
                     'Momento': Momento,
                     'No. de personas': Personas
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('costos_pred.csv', encoding='latin-1')
X = datos.drop(columns='Costo')
y = datos['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613797)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

st.subheader('Cálculo del costo')
st.write('El costo de la actividad será de: ', prediccion)
