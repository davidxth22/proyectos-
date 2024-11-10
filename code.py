import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.stattools import adfuller
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
#-----------------------------------------------------------------------------------------------------------------------------------------
# streamlit run code.py


st.set_page_config(layout="wide")
st.write("""
# Analisís de datos empresa turismo
""")


#Datos
datos = pd.read_csv("datos.csv")

#-----------------------------------------------------------------------------------------------------------------------------------------
#Kmeans code
X = datos[['N° Noches', 'N° Pasajeros', 'Precio x Noche','Mascota','Pago en USD']]
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak")
kmeans = KMeans(n_clusters=4, random_state=0,n_init=10)
X.loc[:, 'Cluster']= kmeans.fit_predict(X)
X.loc[:, 'Cluster'] = X.loc[:, 'Cluster'].astype(str)


#grafico cantidad de clientes

counts = X["Cluster"].value_counts().sort_index()
counts.index = ["Cliente 1","Cliente 2","Cliente 3","Cliente 4"]
fig1 = plt.figure(figsize=(12, 6))
sns.barplot(x=counts.index, y=counts.values);
plt.title("Cantidad de cada cliente")
plt.ylabel("Cantidad")
plt.xlabel("Tipo de cliente")




#Grafico promedio clientes
cluster_means =X.groupby('Cluster').mean()

# Definir el número de filas (clusters) y columnas (variables)
num_clusters = len(cluster_means.index)
num_vars = len(cluster_means.columns)

# Obtener los máximos y mínimos de cada variable para mantener la escala
y_lims = {var: (cluster_means[var].min() * 0.9, cluster_means[var].max() * 1.1) for var in cluster_means.columns}

# Crear figura y subgráficos con tamaño adecuado (3 filas, 5 columnas)
fig2, axes = plt.subplots(nrows=num_clusters, ncols=num_vars, figsize=(12, 13), constrained_layout=True)

# Colores distintivos por clúster (por fila)
row_colors = ['skyblue', 'lightgreen', 'salmon','yellow']  # Añadir más colores si hay más clusters

# Iterar sobre los clusters y variables para rellenar cada subgráfico
h=1
for i, cluster in enumerate(cluster_means.index):
    for j, var in enumerate(cluster_means.columns):
        # Graficar cada variable de cada clúster en la posición correspondiente
        axes[i, j].bar([f'Cliente {h}'], [cluster_means.loc[cluster, var]], color=row_colors[i % len(row_colors)], edgecolor='black')
        axes[i, j].set_title(f'{var}', fontsize=14)
        axes[i, j].set_ylabel('Promedio', fontsize=9)
        #axes[i, j].set_xlabel('Clúster', fontsize=9)
        # Establecer los mismos límites de Y en cada columna (variable) y mostrar diferencias
        axes[i, j].set_ylim(y_lims[var])
        
        
        # Aumentar tamaño de etiquetas del eje Y para mejor visibilidad
        axes[i, j].tick_params(axis='y', labelsize=9)
    h+=1
# Ajustar el diseño para evitar solapamiento de títulos y etiquetas
plt.suptitle('Promedio de Variables por Cliente', fontsize=16, weight='bold')

#-----------------------------------------------------------------------------------------------------------------------------------------
#Estadistica descriptiva
columns_to_plot_pie = [
    'Año', 
    'Asset Code', 
    'Mascota', 
    'Canal de Venta', 
    'Medio de Pago', 
    'Boleta/Factura', 
    'Pago en USD'
]

# Configuración de estilo de Seaborn
sns.set_theme(style="whitegrid")
fig3, axes = plt.subplots(4, 3, figsize=(18, 20))  # Tamaño de la figura ajustado

# Crear gráficos de torta para cada columna
for i, column in enumerate(columns_to_plot_pie):
    # Seleccionar el subplot adecuado en la cuadrícula
    row, col = divmod(i, 3)
    ax = axes[row, col]
    
    # Contar la cantidad de observaciones por cada categoría en la columna
    counts = datos[column].value_counts()
    
    # Crear el gráfico de torta
    if column == "Mascota":
        cc = ["Mascota","Sin Mascota"]
        ax.pie(counts, labels=cc, autopct='%1.1f%%', startangle=90)
    elif column == "Pago en USD":
        cc = ["Paga en USD","No paga en USD"]
        ax.pie(counts, labels=cc, autopct='%1.1f%%', startangle=90)
    else:
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    
    ax.set_title(f'Distribución de Observaciones por {column}')
    ax.axis('equal')  # Para que el gráfico de torta sea un círculo

# Ocultar subplots vacíos si hay menos de 12 columnas
for j in range(len(columns_to_plot_pie), 12):
    fig3.delaxes(axes.flatten()[j])

# Ajustar el espacio entre subplots
plt.tight_layout()
plt.show()








#-----------------------------------------------------------------------------------------------------------------------------------------
#Serie

df_serie = datos[['Año','Mes','Precio x Noche']].copy()
df_serie = datos[['Año','Mes','Precio x Noche']].copy()
meses = {
    'Enero': '01', 'Febrero': '02', 'Marzo': '03', 'Abril': '04',
    'Mayo': '05', 'Junio': '06', 'Julio': '07', 'Agosto': '08',
    'Septiembre': '09', 'Octubre': '10', 'Noviembre': '11', 'Diciembre': '12'
}
df_serie['Mes'] = df_serie['Mes'].str.split('_').str[0].map(meses)
df_serie['T'] = pd.to_datetime(df_serie['Año'].astype(str) + '-' + df_serie['Mes'] + '-01', format='%Y-%m-%d')
df_serie.drop(["Mes","Año"],inplace = True,axis = 1)
serie_mes = df_serie.groupby('T')['Precio x Noche'].mean().reset_index()
serie_mes['T'] = pd.to_datetime(serie_mes['T'])  # Convertir a datetime
serie_mes.set_index('T', inplace=True)
#separación train - test
fecha_inicio = '2022-01-01'
fecha_fin = '2024-03-01'
# Filtrar el DataFrame por el rango de fechas
train = serie_mes.loc[(serie_mes.index >= fecha_inicio) & (serie_mes.index <= fecha_fin)]
fecha_inicio = '2024-04-01'
fecha_fin = '2024-09-01'
test = serie_mes.loc[(serie_mes.index >= fecha_inicio) & (serie_mes.index <= fecha_fin)]

p,d,q,P,D,Q = (2, 0, 1, 1, 0, 1)
modelo_arima = ARIMA(train,order=(p, d, q), seasonal_order=(P,D,Q,12))
modelo_fit = modelo_arima.fit()
pred_start = test.index[0]
pred_end = test.index[-1]
forecast = modelo_fit.get_forecast(steps=len(test))
pred = forecast.predicted_mean
intervalos_confianza = forecast.conf_int()
#mse = mean_squared_error(test, pred)
pred = pd.DataFrame(pred)

fig4 = plt.figure(figsize=(12, 6))
sns.lineplot(data=train, x='T', y='Precio x Noche', marker='o',label='Entrenamiento')
sns.lineplot(data=test, x='T', y='Precio x Noche', marker='o',label='Prueba')
sns.lineplot(data=pred, x=pred.index, y='predicted_mean', marker='o',label='Predicción')

plt.fill_between(pred.index, 
                 intervalos_confianza.iloc[:, 0],  # Límite inferior
                 intervalos_confianza.iloc[:, 1],  # Límite superior
                 color='gray', alpha=0.2, label='Intervalo de confianza')

plt.title('Promedio Precio x Noche por Mes')
plt.xlabel('Año-Mes')
plt.ylabel('Promedio de Precio x Noche')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()













#-----------------------------------------------------------------------------------------------------------------------------------------
#graficos

#col1, col2 = st.columns([2, 1])
col1, col2 = st.columns([1,2]) 
col3 = st.container()  
with col1:
    st.write("### Clientes")
    st.pyplot(fig1)
    st.pyplot(fig2)

with col2:
    st.write("### Datos")
    st.pyplot(fig3) 

with col3:
   st.write("### Serie")
   st.pyplot(fig4) 





#st.pyplot(fig1)  # Primer gráfico
#st.pyplot(fig2)  # Segundo gráfico


#col1, col2, col3 = st.columns(3)

#with col1:
#    st.pyplot(fig1)
#    st.write("Este es el gráfico del seno.")


#with col2:
#    st.pyplot(fig1)
#    st.write("Este es el gráfico del seno.")


#with col3:
#    st.pyplot(fig1)
#    st.write("Este es el gráfico del seno.")





