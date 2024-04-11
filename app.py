# Librerías necesarias para nuestro análisis:
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly_express as px
import plotly.graph_objects as go
import warnings

# Modelos clasificatorios:
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Modelos de predicción para entrenar el modelo.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Suprimir los warnings
warnings.filterwarnings('ignore')

#-----CONFIGURACION APP------

st.set_page_config(page_title="ENFERMEDAD ALZHEIMER, CAUSAS Y DIÁGNOSTICO FINAL", layout="centered")

#------CARGA ARCHIVO CSV--------

# Cargar los datos
df = pd.read_csv('oasis_longitudinal.csv')

#----CAMBIOS DEL DATASET----

# Comprobar si hay columnas con valores faltantes
missing_values_in_columns = df.isnull().sum()
print(missing_values_in_columns)

# Lista de columnas con valores faltantes
columns =  ['SES', 'MMSE']

# Rellenar valores faltantes con la media porque son datos numéricos
for column in columns:
    df[column].fillna(value=df[column].mean(), inplace=True)

# Eliminar columnas 'Subject ID', 'MRI ID', 'Visit', 'MR Delay', 'Hand', 'ASF' ya que no aportan información relevante para nuestro estudio. 
df.drop(columns=['Subject ID', 'MRI ID', 'Visit', 'MR Delay', 'Hand', 'ASF'], inplace=True)

# Cambiamos el nombre de la columna M/F a 'Gender' para que sea más reconocible.
df.rename(columns={'M/F': 'Gender'}, inplace=True)

# Pasar datos de la columna ´SES´y ´MMSE´ a número entero
df['SES'] = df['SES'].astype(int)
df['MMSE'] = df['MMSE'].astype(int)

# Dividir la columna 'EDUC' en grupos según etapas educativas
mapping = {}
for i in range(6, 24):
    if 6 <= i <= 8:
        mapping[i] = '1'
    elif 9 <= i <= 11:
        mapping[i] = '2'
    elif 12 <= i <= 14:
        mapping[i] = '3'
    elif 15 <= i <= 17:
        mapping[i] = '4'
    elif 18 <= i <= 23:
        mapping[i] = '5'

# Replace the values in the educ column with the groups
df['EDUC'] = df['EDUC'].replace(mapping)

# Cambiar columna 'Gender' a datos binarios:
def funcion1(fila):
    if fila['Gender'] == 'F':
        return 1
    elif fila['Gender'] == 'M':
        return 0
    else:
        return np.nan  

df['Gender'] = df.apply(funcion1, axis=1)

# Creamos una copia del dataset original sin modificar por otro ya modificado para comenzar con nuestro ánalisis:
df_original = df.copy()

# Guardar el DataFrame df_original como un archivo CSV y lo cargamos
df_original.to_csv('df_original.csv', index=False)

df_original = pd.read_csv('df_original.csv')

# Cambiamos columna de 'Group' a los tres estados de diagnóstico:
mapping = {'Nondemented': 0, 'Demented': 1, 'Converted': 2}  # Define your mapping here
df_original['Group'] = df['Group'].replace(mapping)

#---COMENZAR APP----

# Crear un sidebar para las opciones de filtrado
st.sidebar.header('Opciones de filtrado')

# Crear un selector para seleccionar columnas
columns = st.sidebar.multiselect('Selecciona las columnas que quieres visualizar', ['Group', 'Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV'])

# Mostrar las columnas seleccionadas en la aplicación
st.dataframe(df[columns])

# Agregar un subtítulo con formato
st.markdown("## Aplicación sobre el análisis y predicción de la enfermedad del Alzheimer")

#Información sobre la enfermedad a modo introductorio.

st.header("ESTUDIO SOBRE EL ALZHEIMER Y SUS POSIBLES FACTORES DE RIESGO PARA SU DIAGNOSTICO")

st.markdown(""""
El Alzheimer es la forma más común de demencia, un término general que se aplica a la pérdida de memoria y otras habilidades cognitivas que interfieren con la vida cotidiana. La enfermedad de Alzheimer es responsable de entre un 60 y un 80 por ciento de los casos de demencia. El Alzheimer no es una característica normal del envejecimiento. El factor de riesgo conocido más importante es el aumento de la edad, y la mayoría de las personas con Alzheimer son mayores de 65 años. Pero el Alzheimer no es solo una enfermedad de la vejez. 

A medida que la enfermedad de Alzheimer progresa, las personas tienen una mayor pérdida de memoria y otras dificultades cognitivas

En el caso de un diagnóstico de la enfermedad de Alzheimer, comenzar el tratamiento en las primeras etapas del proceso de la enfermedad puede ayudar a preservar el funcionamiento diario durante cierto tiempo. Un diagnóstico temprano también ayuda a las familias a planificar para el futuro.

DATASET: oasis_longitudinal.csv: Estos datos constan de más de 373 muestras correspondientes a exploraciones MRI de 150 personas de entre 60 y 98 años. "Longitudinal" se refiere al hecho de que los datos fueron recopilados durante varias revisiones de los sujetos en diferentes momentos.

SOURCE:  Open Access Series of Imaging Studies (OASIS) 
""")
# Agregar una imagen a la aplicación de Streamlit
st.image('concepto-demencia-perdida-memoria-alzheimer-creado-tecnologia-ia-generativa-980x553.jpg')

st.markdown("""""
PROPÓSITO DEL ANÁLISIS: Según una serie de condiciones o parámetros, podemos predecir si una persona es diagnosticada o no con la enfermedad de Alzheimer y realizamos un análisis donde vemos qué relaciones y conclusiones podemos deducir de estos indicadores.
""""")

# Agregar texto con formato
st.markdown("""
Esta aplicación te permite visualizar y analizar datos sobre la enfermedad del Alzheimer.
Puedes seleccionar las columnas que quieres visualizar en el menú de la izquierda.
""")

# Agregar una lista con formato
st.markdown("""
Características de los datos:

- **Group**: Grupo al que pertenece el paciente (No demente, Demente o Convertido(Pasa de un falso diagnóstico a tener la enfermedad)
- **Gender**: Género del paciente (masculino, femenino)
- **Age**: Edad del paciente
- **EDUC**: Nivel de educación del pacienteAlmacena el grado de instrucción. Clasificado entre las categorías: 1: secundaria incompleta, 2: secundaria completa, 3: universidad incompleta, 4: universidad completa ,5: post-bachiller        
- **SES**: Estatus socioeconómico del paciente. Clasificado entre las categorías de 1 (más alto status) to 5 ( más bajo status)
- **MMSE**: Almacena el puntaje en el test MMSE, que mide la funcion cognitiva. Rangos de 0 (Peor) a 30 (Mejor), puntajes menores a 24 sugiere función 
- **CDR**: Contiene el puntaje en el test CDR, que evalúa el grado de deterioro cognitivo. 0 : Sin demencia, 0.5 : Demencia muy leve, 1 : Demencia leve, 2 : Demencia moderada.     
- **eTIV**: Volumen intracraneal estimado
- **nWBV**: Volumen cerebral normalizado
""")

# Mostrar las columnas seleccionadas en la aplicación
st.dataframe(df[columns])

#-- COMENZAMOS CON LOS PRIMEROS GRAFICOS----

# Calcular estadísticas descriptivas (hacer un describe del dataset)
desc = df_original.describe()

# Mostrar las estadísticas descriptivas en la aplicación de Streamlit
st.dataframe(desc)

# Agregar descripción
st.write("Aquí observamos un resumen de las estadísticas descriptivas claves de todo el dataset. Como primera conclusión, se observa que todas las columnas cuentan con 373 filas, por lo que no tenemos datos faltantes. La media de más del 0,5 en el género tiende hacia el grupo 1 dónde demuestra mayor número de mujeres. La edad de los pacientes varía desde 6 hasta 98 años, cn un promedio de 77. El MMSE evalúa la función cognitiva, que tiene un promedio de aproximadamente 0.85. El CDR mide la gravedad de la demencia dónde hay valores que van variando desde 0 hasta 2. La desviación típica (std), muestra el valor más alto y el más bajo y la diferencia de volatibilidad entre ellos.\n\nEn resumen, este conjunto de datos proporciona información sobre características clínicas y medidas de volumen cerebral en pacientes con Alzheimer. El eTIV y el nWBV son especialmente relevantes para comprender la relación entre la atrofia cerebral y la progresión de la enfermedad.")

st.header("ANÁLISIS EXPLORATORIO (EDA)")

#Primer grafico interactivo pie chart
df_grouped = df.groupby(['Gender', 'Group']).size().reset_index(name='counts')

fig = go.Figure(data=[go.Pie(labels=df_grouped['Group'], values=df_grouped['counts'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(
    title_text="Total de pacientes por diagnóstico",
    # Añadir anotaciones en el centro de el pie chart
    annotations=[dict(text='Group', x=0.5, y=0.5, font_size=20, showarrow=False)]
)

# Mostrar el gráfico en la aplicación de Streamlit
st.plotly_chart(fig)

st.write("Este gráfico muestra el total de pacientes por diagnóstico. El total de datos demuestra como en la mayoría de los 150 pacientes en las 373 muestras realiazadas para este estudio acaban teniendo un buen diágnostico, frente al cercano valor de personas si dementes. Los resultados del diágnostico serían: Nondemented': 0, 'Demented': 1, 'Converted': 2")


# Otro grafico de 'Género' es una variable binaria donde 0 representa 'Hombre' y 1 representa 'Mujer'
df_grouped = df.groupby(['Gender', 'Group']).size().reset_index(name='counts')

# Pie chart 
fig, axs = plt.subplots(1, 2, figsize=(12,6))

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

for i, gender in enumerate(df_grouped['Gender'].unique()):
    df_gender = df_grouped[df_grouped['Gender'] == gender]
    gender_label = 'Male' if gender == 0 else 'Female'  
    axs[i].pie(df_gender['counts'], labels = df_gender['Group'], autopct='%1.1f%%', colors=colors)
    axs[i].set_title(f'Distribución de la enfermedad por género: {gender_label}')  
    axs[i].add_artist(plt.Circle((0,0),0.70,fc='white'))

# Mostrar el gráfico en la aplicación de Streamlit
st.pyplot(fig)

# Agregar texto debajo del gráfico
st.write("En esta muestra cabe destacar como el género masculino supera en casi el doble el total de afectados por la demencia en comparación con las mujeres. Por otro lado, el grupo de converted (personas que en un principio dieron un resultado negativo pero luego demostraron tener demencia) muestra un porcentaje m'as similar y es pequeño en ambos géneros.")

# Crear el gráfico de dispersión
fig, ax = plt.subplots()
ax.scatter(df['Group'], df_original['Age'])

# Mostrar el gráfico en la aplicación de Streamlit
st.pyplot(fig)

# Agregar el texto
st.write("""
Este gráfico de dispersión compara la posible relación de la edad con la categoría de diagnóstico. Se observa que las personas no dementes están muy concentrados entre los 60 a 95 años y no muestran desajustes. Las personas dementes muestran que más variabilidad, concentrándose más en el extremo superior, según avanza la edad. Los convertidos tienen mayor dispersión, por lo que su estado puede demostrar muchos cambios significativos. El rango de 75 a 90 muestra una continuidad común en todos los grupos.
""")

#---- CAMBIOS A DF_ORIGINAL---

# Cambiamos columna de 'Group' a los tres estados de diagnóstico:
mapping = {'Nondemented': 0, 'Demented': 1, 'Converted': 2}  # Define your mapping here
df_original['Group'] = df['Group'].replace(mapping)

# Dividir la edad en estado de Alzheimer precoz (menor de 60) y avanzado (mayor de 65):
def funcion2(fila):
    if fila['Age'] < 65:
        return 0
    elif fila['Age'] >= 65:
        return 1
    else:
        return np.nan
    
df_original['Age'] = df_original.apply(funcion2, axis=1)

# Dividir el MMSE (que mide el nivel de funcion cognitiva) en  mayores de 24 (sugiere  función normal) de menores de ese valor (nivel cognictivo malo).
def divide_MMSE(fila):
    if fila['MMSE'] > 24:
        return 1  # sugiere función normal
    elif fila['MMSE'] <= 24:
        return 0  # nivel cognitivo malo
    else:
        return np.nan

df_original['MMSE'] = df_original.apply(divide_MMSE, axis=1)

# Colocar la columna 'Group' al principio
cols = list(df_original.columns)
cols.remove('Group')
cols.append('Group')
df_original = df_original[cols]

# Crear un nuevo dataset con los cambios que será el definitivo para realizar las siguientes analiticas y el estudio de ML.
clean_data = df_original.copy()

clean_data.to_csv('clean_data.csv', index=False)

df_clean = pd.read_csv('clean_data.csv')

#----NUEVOS GRAFICOS CON EL DATASET DEFINITIVO CREADO (CLEAN_DATA)

#Grafico

plt.figure(figsize=(12,8))
sns.countplot(x='EDUC', hue='Group', data=df_clean, order=sorted(df_clean['EDUC'].unique()), palette=['green', 'yellow'])

plt.title('Distribución de nivel educativo según grupo de diagnóstico')
plt.xlabel('EDUC: 1:secundaria incompleta, 2:secundaria completa, 3:universidad incompleta, 4:universidad completa ,5:post-bachiller')
plt.ylabel('Count')

# Get the legend and set the labels
legend = plt.legend(title='Group')
legend.texts[0].set_text('Nondemented')
legend.texts[1].set_text('Demented')
legend.texts[2].set_text('Converted')

# Mostrar el primer gráfico en la aplicación de Streamlit
st.pyplot(plt.gcf())
plt.clf()

# Segundo gráfico
plt.figure(figsize=(12,8))
sns.barplot(x='EDUC', y='MMSE', data=df_clean, color='yellow', errcolor=None)

plt.title('Comparación del nivel de función cognictiva por nivel educativo')
plt.xlabel('EDUC: 1:secundaria incompleta, 2:secundaria completa, 3:universidad incompleta, 4:universidad completa ,5:post-bachiller')
plt.ylabel('MMSE')

# Mostrar el segundo gráfico en la aplicación de Streamlit
st.pyplot(plt.gcf())
plt.clf()
  
  # Agregar el texto a este grafico
st.write("""
El grado educativo no es determinante para predecir un diagnóstico final de demencia, pero los estudios demuestran que a mayor nivel educativo(universidad o post-grado) la probabilidad de sufrir la enfermedad es mucho menor. Estos datos también pueden influir en el desarrollo normal del cerebro en personas jóvenes, la etapa de vida más habitual dónde se realiza este nivel de estudios.

Según estudios, aunque el cerebro alcanza su mayor tamaño en la adolescencia temprana, los años de la adolescencia sirven para afinar su funcionamiento. El cerebro termina de desarrollarse y de madurar entre los 25 y los 30 años (de ahí que esos datos alcancen ese resultado).

La función cognictiva sigue una distribución de valores bastante favorable y normal, sólo sufriendo un ligero descenso en el grupo educativo 4, pero nunca por debajo del nivel cognictivo anormal de 24.
""")

#Imagen
st.image('1664484802109.jpg')

#Otro grafico
fig = px.box(df, x='Age', y='nWBV')

fig.update_layout(
title='Comparación de nWBV (volumen total del cerebro normalizado) por edad',
xaxis_title='Age',
 yaxis_title='nWBV'
)

# Mostrar el gráfico en la aplicación de Streamlit
st.plotly_chart(fig)

# Agregar el texto a la aplicación de Streamlit
st.write("""
Aquí se observa un indicador clave que puede resulta crucial en nuestro estudio, dónde se muestra cómo a medida que la edad avanza en un sujeto, el tamaño del cerebro disminuye considerablemente de volmen, habiendo un deterioro muy marcado a partir de los 70 años. El nWBV resulta un indicador valioso para está deducción.
""")
# Agregar la imagen a la aplicación de Streamlit
st.image('brain_slices_alzheimers_spanish_version.jpg')

# Nuevo grafico

plt.figure(figsize=(10,6))
sns.boxplot(x='Age', y='eTIV', data=df_clean, hue='Age', palette="Set3")
sns.stripplot(x='Age', y='eTIV', data=df_clean, color=".25", jitter=0.2)

# Crear leyenda a edad
precoz = mpatches.Patch(color=sns.color_palette("Set3")[0], label='0: < 65 Alzheimer precoz')
avanzado = mpatches.Patch(color=sns.color_palette("Set3")[1], label='1: > 65 Alzheimer avanzado')
plt.legend(handles=[precoz, avanzado])

plt.title('Distribución de eTIV (Volumen intracraneal total estimado) según etapa del Alzheimer')
plt.xlabel('Age')
plt.ylabel('eTIV')

# Mostrar el gráfico en la aplicación de Streamlit
st.pyplot(plt.gcf())
plt.clf()

# Agregar el texto a la aplicación de Streamlit
st.write("""
Analizamos el volumen intracraneal total estimado (eTIV) en dos grupos de pacientes con Alzheimer, diferenciados por la etapa de la enfermedad, dónde puede observarse que el grupo 1 con la enfermedad en estado avanzado, muestra una concentración intensa de valores, asi que es dónde se debería poner el foco en pacientes que muestran más riesgo. A medida que avanza la enfermedad de Alzheimer, se observa una disminución gradual en el eTIV.
Según estudios, esta disminución está relacionada con la atrofia cerebral, que en etapas tempranas, el eTIV puede estar relativamente preservado, pero a medida que progresa la enfermedad, se produce una atrofia significativa que afecta principalmente la corteza cerebral.                                 
El grupo 0 muestra datos muy dispersos y sin seguir una tendecia clara.
Sin duda, este seguimiento del eTIV a lo largo del tiempo puede ayudar a los médicos a evaluar la progresión de la enfermedad y ajustar el manejo clínico.
""")

#Grafico histplot de distribucion normal
fig, axes = plt.subplots(2, 2, figsize=(14,10))

sns.histplot(ax=axes[0, 0], data=df, x='nWBV', kde=True)
axes[0, 0].set_title('Distribución de nWBV')

sns.histplot(ax=axes[0, 1], data=df, x='MMSE', kde=True)
axes[0, 1].set_title('Distribución de MMSE')

sns.histplot(ax=axes[1, 0], data=df, x='CDR', kde=True)
axes[1, 0].set_title('Distribución de CDR')

sns.histplot(ax=axes[1, 1], data=df, x='EDUC', kde=True)
axes[1, 1].set_title('Distribución de EDUC')

plt.tight_layout()

# Mostrar los gráficos en la aplicación de Streamlit
st.pyplot(fig)
plt.clf()

# Agregar el texto a la aplicación de Streamlit
st.write("""
Distribución de algunas variables asociadas a los cuadros de Demencia:

Se observa que el volumen total del cerebro normalizado sigue una distribución aparentemente normal con una mediana de 0.74; en el test MMSE hay una proporción alta que puntúan por encima de 24 en el test; respecto al CDR, de los que presentan algún grado de demencia, se observan mayor frecuencia en demencia muy leve (0.5); y en cuanto al grado de educación, con mayor frecuencia los sujetos presentaron un nivel educativo de universidad incompleta (3).
""")


#Poner un heatmap

plt.figure(figsize=(10,8))
sns.heatmap(df_original.corr(), annot=True, cmap='coolwarm')

plt.title('Mapa de calor de correlaciones')
st.pyplot(plt)

st.write("""
Con los valores de los 416 sujetos en el dataset, las correlaciones en este mapa de calor señalan que de las variables de interés aquellas que presentan una mayor correlación son tres: nWBV-MMSE, CDR-Group y nWBV-Gender, dónde toda la atención cae en CDR-Group (el único grupo con la correlación más fuerte y cercana a 1.) En general, el dataset es poco correlativo y muy disperso entre sí. Esto indica que el Alzheimer puede ser influenciado por muchos factores distintos. 
Que sólo haya un único valor cercano a 1 (0.57) y una proporción decente de valores cercanos a cero demuestra que el dataset es de calidad y de buena salud estadística para este tipo de estudio. 
""")

#------COEFICIENTES-------

st.header("COEFICIENTES PARA ESTUDIAR LA CALIDAD DE LOS DATOS DEL DATASET EN TÉRMINOS DE ANÁLISIS Y PREDICCIÓN")

st.code(
"#Vamos a probar el coeficiente Kappa\n"
"x = df_clean[['Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV']]\n" 
"y = df_clean['Group']\n"
"# Escalar los datos\n"
"scaler = StandardScaler()\n"
"x_scaled = scaler.fit_transform(x)\n"
"x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25)\n"
"# Aumentar el número de iteraciones\n"
"modelo = LogisticRegression(max_iter=10000)\n"
"modelo.fit(x_train, y_train)\n"
"y_pred = modelo.predict(x_test)\n"
"# Calcular el coeficiente Kappa\n"
"kappa = cohen_kappa_score(y_test, y_pred)\n"
"# Mostrar el coeficiente Kappa en la aplicación de Streamlit\n"
"st.write(f'El coeficiente kappa es: {kappa*100:.2f}%')\n", language='python')

st.write("COEFICIENTE KAPPA. Conclusión: 0.7835 demuestra un porcentaje aceptable, por lo que sugiere que los clasificadores están alineados en sus predicciones. Valores cercanos a 1 señalan mayor concordancia que la esperada por azar, pero estos resultados también pueden llegar a ser muy aleatorios.")

st.code(
"distancia = cosine_distances([vector1], [vector2])[0, 0]\n"
"st.write(f'Distancia del coseno: {distancia}')\n", language='python')

st.write("DISTANCIA COSENO: 1.913671443531939e-10 El resultado obtenido por la distancia del coseno está en notación científica, por lo que el valor es bastante pequeño, siendo la distancia entre los vectores cercana y eso indica que los registros se parecen bastante entre sí")

#  Calculamos la medida F1
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import streamlit as st

x = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Crear un pipeline que primero escala los datos y luego aplica la Regresión Logística.
# También aumenta el número máximo de iteraciones permitidas en el modelo de Regresión Logística a 1000.
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo.fit(x_train, y_train)
y_pred = modelo.predict(x_test)
# Generar classification report
report = classification_report(y_test, y_pred)
# Mostrar el classification report en la aplicación de Streamlit
st.text(report)
st.write("MEDIDA F1: para evaluar el rendimiento del modelo.  Con una precisión que ronda del 85% al 98%, indiga un porcentaje de predicciones correcto. La exhaustividad del recall es de hasta poder alcanzar del 98% para la clase 0 y 100% para la clase 1. Esto indica que el modelo identifica correctamente la mayoría de los casos positivos.")


#---IMPLEMENTACION DEL MODELO MACHINE LEARNING-----

st.header("IMPLEMENTACION DE MODELOS MACHINE LEARNING")

st.write("He usado un modelo ´Lazy´ kNN y uno complicado de Ramdon Forest dónde compruebo la calidad del dataset para predecir.")

st.write("Intento 1: kNN predeterminado con la distancia de Euclídea. La distancia euclidiana es una métrica de distancia que se utiliza para datos continuos.")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

x = df_clean.iloc[:, :-1] # Seleccionamos todas las columnas excepto la última
y = df_clean.iloc[:, -1] # Elegimos sólo la última columna
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
knn = KNeighborsClassifier(n_neighbors=30) # Ajustamos el parámetro 'k'.
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
# Mostrar la precisión del modelo en la aplicación de Streamlit
st.write(f'Precisión: {accuracy}')


#Grafico de valor k optimo

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

x = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Range de 'k' a probar
k_range = list(range(1, 31))  # Convert range to list

# Lista para mostrar la precision de 'k'
accuracies = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Encontrar la 'k' con mayor precision
k_optimal = k_range[accuracies.index(max(accuracies))]

# Mostrar grafico de 'k' optimo con pandas
df = pd.DataFrame({'k': k_range, 'accuracy': accuracies})
df.set_index('k').plot(grid=True, figsize=(10, 6))

# Mostrar el gráfico en la aplicación de Streamlit
st.pyplot(plt.gcf())
plt.clf()

# Mostrar el valor óptimo de k y su precisión correspondiente en la aplicación de Streamlit
st.write(f'El valor óptimo de k es: {k_optimal} con una precisión del {max(accuracies)*100:.2f}%')

st.code("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

x = df_clean.iloc[:, :6]
y = df_clean['Group']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Escogemos k=10 porque es el óptimo.
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
confm = confusion_matrix(y_test, y_pred)
st.write(confm)
cm = confusion_matrix(y_test, y_pred)
# Visualizar matriz de confusión
sns.heatmap(cm, annot=True, fmt='d')
# Mostrar el gráfico en la aplicación de Streamlit
st.pyplot(plt.gcf())
plt.clf()
""", language='python')

# Ejecuta el código para generar el gráfico
x = df_clean.iloc[:, :6]
y = df_clean['Group']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
confm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
st.pyplot(plt.gcf())
plt.clf()

st.markdown("Conclusiones de los resultados\n\nEn primer lugar descartamos que el dataset sea sensible al ruido  ya que el 'k' óptimo seleccionado entre 1-10 siempre da valores próximos al 60%.\n\nLa matriz de confusión es excelente ya que no refleja falsos positivos ni falsos negativos.")


st.markdown("VAMOS A PROBAR UN MODELO DE LENGUAJE SUPERVISADO MÁS PRECISO COMO RAMDON FOREST PARA COMPROBAR LA PRECISIÓN DEL DATASET")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st

#Modelo Random Forest
x = df_clean.iloc[:, :-1]  # Select all columns except the last one as features
y = df_clean.iloc[:, -1]  # Select the last column as the target variable

# Split the data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create a Random Forest Classifier
clf = RandomForestClassifier()

# Train the classifier with the training data
clf.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
st.write(f'Precisión: {accuracy*100:.2f}%')

# Print the classification report
report = classification_report(y_test, y_pred)
st.text(report)

st.markdown("El mejor algoritmo para implementar nuestro algoritmo predictivo con una precisión cerca o superior al 90% es el Random Forest.")

#Hacer la prediccion de la app

# Selecciona las primeras 8 columnas de df_clean
x = df_clean.iloc[:, :8]

# Asegúrate de que y sigue siendo la misma columna 'Group'
y = df_clean['Group']

# Crear un Random Forest Classifier
clf = RandomForestClassifier()
# Asumiendo que ya has entrenado tu modelo con los datos
clf.fit(x_train, y_train)

# Crear selectboxes para los parámetros del usuario
param1 = st.selectbox('Selecciona tu género:', options=['Masculino', 'Femenino'])
param2 = st.selectbox('Selecciona tu edad entre estos rangos:', options=['-60', '60-65', '65-98'])
param3 = st.selectbox('Selecciona tu nivel educativo entre estas opciones:', options=['Hasta secundaria completa finalizada', 'Estudios medios o universidad completa', 'Estudios post-universitarios'])
param4 = st.selectbox('Selecciona entre estas opciones el que consideras tu nivel socioeconómico en un rango del 0 al 5, siendo 1 muy bajo status y cinco muy elevado:', options=['1', '3', '5'])
param5 = st.number_input('Ingresa un valor entre 4 a 30 dónde consideres evalues tu función cognitiva, considerando rangos de 4 peores, un valor menor a 24 sugiere función anormal y los que están cerca del 30 dentro de los valores apropiados', min_value=4, max_value=30)

# Añadir 3 parámetros numéricos adicionales
param6 = st.number_input('Ingresa un valor de 0 a 2 según consideres tu grado cognictivo con la enfermedad:, siendo 0 : Sin demencia, 0.5 : Demencia muy leve, 1 : Demencia leve, 2 : Demencia avanzada:', min_value=0, max_value=2) 
param7 = st.selectbox('Selecciona el valor desde 1,106 a 2,004, para considerar tu volumen intracraneal total estimado en mm3 (dónde valores por debajo de 1,600 son un mal indicativo):', options=['1106-1600', '1600-2004'])
param8 = st.selectbox('Selecciona un valor para el volumen total del cerebro normalizado dónde a mayour cantidad y hasta 0.729 tóxeles en un volumen de cerebro sano:', options=['0.644', '0.729', '0.837'])

# Mapear los parámetros a números
gender_map = {'Masculino': 1, 'Femenino': 0}  # Supongamos que 'Masculino' tiene un mayor riesgo
age_map = {'-60': 0, '60-65': 2, '65+': 3, '65-98': 4}
education_map = {'Hasta secundaria completa finalizada': 0, 'Estudios medios o universidad completa': 1, 'Estudios post-universitarios': 2}  # Supongamos que el riesgo disminuye con la educación
socioeconomic_map = {'1': 1, '3': 3, '5': 5}  # Supongamos que el riesgo es mayor para los niveles socioeconómicos más bajos

param1 = gender_map[param1]
param2 = age_map[param2]
param3 = education_map[param3]
param4 = socioeconomic_map[param4]
param6_map = {0: 0, 0.5: 1, 1: 2, 2: 3}
param7_map = {'1106-1600': 1, '1600-2004': 2}  # Supongamos que el riesgo es mayor para los valores más bajos
param8_map = {'0.644': 1, '0.729': 2, '0.837': 3}

# Mapear param7 a un número
param7 = param7_map[param7]

if st.button('¿Podría sufrir demencia?'):
    # Crear un diccionario que mapee los números a las clases
    class_dict = {0: 'Nondemented', 1: 'Demented', 2: 'Converted'}
    # Hacer la predicción con los parámetros del usuario
    prediction = clf.predict([[param1, param2, param3, param4, param5, param6, param7, param8]])
    # Verificar si el modelo hizo una predicción
    if len(prediction) > 0:
        # Verificar si la predicción está en class_dict
        if prediction[0] in class_dict:
            # Obtener la clase correspondiente a la predicción
            predicted_class = class_dict[prediction[0]]
            # Mostrar la predicción
            st.write(f'La predicción del modelo es: {predicted_class}')
        else:
            st.write(f'La predicción del modelo es un número desconocido: {prediction[0]}')
    else:
        st.write('El modelo no hizo ninguna predicción.')

