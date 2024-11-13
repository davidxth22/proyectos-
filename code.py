import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
#-----------------------------------------------------------------------------------------------------------------------------------------
# streamlit run code.py


st.set_page_config(layout="wide")

# Título del proyecto
st.write("""
# Prototipo de Análisis de Datos para una Empresa de Turismo
""")

# Descripción del proyecto
st.write("""
Este prototipo forma parte del proyecto final para el curso de Proyectos Estadísticos de la **Universidad Técnica Federico Santa María**. 
Se realizó en el segundo semestre del año 2024 con el objetivo de aplicar técnicas de análisis de datos en el ámbito del turismo.
""")

# Información de los estudiantes
st.write("""
**Estudiantes**: Diego Astaburuaga y David Rivas
""")


opt = st.radio("Sección :",["Explicación de variables y estructura de datos", 
                            "Análisis exploratorio de datos",
                            "Segmentación de clientes",
                            "Análisis de series temporales"])

st.write("### " + opt)





#-----------------------------------------------------------------------------------------------------------------------------------------
#Datos
datos = pd.read_csv("datos.csv") # Punto de mejora a futuro: Hacer que este code dependa de datos.xlsx y haga las transformaciones necesarias para obtener datos.csv




#-----------------------------------------------------------------------------------------------------------------------------------------
#Estadistica descriptiva

@st.cache_data
def barplot_mes(datos):
    # Asegurarse de que la columna "Mes" sea un tipo categórico con un orden específico
    meses_ordenados = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    datos['Mes'] = pd.Categorical(datos['Mes'], categories=meses_ordenados, ordered=True)
    
    # Lista de columnas para gráficos de barras
    columns_to_plot_bar = [
        'Mes',
        'N° Noches', 
        'N° Pasajeros',
        'Periodo'
    ]
    
    # Crear una grilla 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Crear gráficos de barras para cada columna
    for i, column in enumerate(columns_to_plot_bar):
        # Contar la cantidad de observaciones por cada categoría en la columna
        counts = datos[column].value_counts().sort_index()
        
        # Crear el gráfico de barras en la subtrama correspondiente
        sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=axes[i // 2, i % 2])
        
        # Ajustar título, etiquetas, etc.
        axes[i // 2, i % 2].set_title(f'Distribución de Observaciones por {column}', fontsize=16, fontweight='bold')  # Aumentar tamaño del título
        axes[i // 2, i % 2].set_xlabel(column, fontsize=14)  # Aumentar tamaño de la etiqueta del eje x
        axes[i // 2, i % 2].set_ylabel('Cantidad de Observaciones', fontsize=14)  # Aumentar tamaño de la etiqueta del eje y
        axes[i // 2, i % 2].tick_params(axis='x', rotation=45, labelsize=12)  # Aumentar tamaño de las etiquetas del eje x

    # Ajustar el layout para que los gráficos no se superpongan
    plt.tight_layout()
    plt.show()
    
    return fig

# Llamar a la función y almacenar la figura
fig1 = barplot_mes(datos)


@st.cache_data
def barplot(datos):
    # Columnas a graficar
    columns_to_plot_pie = [
        'Año', 
        'Asset Code', 
        'Mascota', 
        'Canal de Venta', 
        'Medio de Pago', 
        'Boleta/Factura', 
        'Pago en USD',
        'Clean Up'
    ]
    
    # Estilo para los gráficos
    sns.set_theme(style="whitegrid")
    
    # Crear subgráficos
    fig3, axes = plt.subplots(2, 4, figsize=(18, 20)) 
    
    # Recorrer las columnas y graficar
    for i, column in enumerate(columns_to_plot_pie):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        counts = datos[column].value_counts()
        
        # Personalizar etiquetas para ciertos casos
        if column == "Mascota":
            cc = ["Mascota", "Sin Mascota"]
            ax.pie(counts, labels=cc, autopct='%1.1f%%', startangle=90)
        elif column == "Pago en USD":
            cc = ["Paga en USD", "No paga en USD"]
            ax.pie(counts, labels=cc, autopct='%1.1f%%', startangle=90)
        else:
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        
        # Configurar título y formato del gráfico
        ax.set_title(f'%Observaciones por {column}')
        ax.axis('equal')  # Asegura que el pie chart sea circular
    
    # Eliminar subgráficos vacíos si los hay
    for j in range(len(columns_to_plot_pie), 8):  # Hay 8 subgráficos posibles, ya que es 2x4
        fig3.delaxes(axes.flatten()[j])

    # Ajustar el diseño y mostrar la figura
    plt.tight_layout()
    plt.show()
    
    return fig3

# Llamar a la función y almacenar la figura
fig2 = barplot(datos)


  


#-----------------------------------------------------------------------------------------------------------------------------------------
#Kmeans code

@st.cache_data
def kmeans(datos):
    #modelo
    X = datos[['N° Noches', 'N° Pasajeros', 'Precio x Noche','Mascota','Pago en USD']]
    warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak")
    kmeans = KMeans(n_clusters=4, random_state=0,n_init=10) 
    X.loc[:, 'Cluster']= kmeans.fit_predict(X)
    X.loc[:, 'Cluster'] = X.loc[:, 'Cluster'].astype(str)
    
    row_colors = ['skyblue', 'lightgreen', 'salmon','yellow']
    
    #grafico cantidad de clientes
    counts = X["Cluster"].value_counts().sort_index()
    counts.index = ["Cliente 1","Cliente 2","Cliente 3","Cliente 4"]
    percentages = (counts / counts.sum()) * 100
    fig1 = plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=percentages.values, palette=row_colors, edgecolor='black');
    plt.title("Porcentaje de cada tipo de cliente", fontsize=16, weight='bold')
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Tipo de cliente")

    #Grafico promedio clientes
    cluster_means =X.groupby('Cluster').mean()
    num_clusters = len(cluster_means.index)
    num_vars = len(cluster_means.columns)
    y_lims = {var: (cluster_means[var].min() * 0.9, cluster_means[var].max() * 1.1) for var in cluster_means.columns}
    fig2, axes = plt.subplots(nrows=num_clusters, ncols=num_vars, figsize=(12, 13), constrained_layout=True) 
    h=1
    for i, cluster in enumerate(cluster_means.index):
        for j, var in enumerate(cluster_means.columns):
            axes[i, j].bar([f'Cliente {h}'], [cluster_means.loc[cluster, var]], color=row_colors[i % len(row_colors)], edgecolor='black')
            axes[i, j].set_title(f'{var}', fontsize=14)
            axes[i, j].set_ylabel('Promedio', fontsize=9)
            axes[i, j].set_ylim(y_lims[var])
            axes[i, j].tick_params(axis='y', labelsize=9)
        h+=1
    plt.suptitle('Promedio de Variables por Cliente', fontsize=16, weight='bold')

    
    
    return X,fig1,fig2

X,fig3,fig4 = kmeans(datos)







#-----------------------------------------------------------------------------------------------------------------------------------------
# Serie de tiempo
@st.cache_data
def serie(datos):
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
    return fig4

fig5 = serie(datos)









#-----------------------------------------------------------------------------------------------------------------------------------------
# Visualizador

if opt == "Explicación de variables y estructura de datos":
    st.markdown("Página en mantención. (Se omite está página en la versión final por temas de diseño)")
    
elif opt == "Análisis exploratorio de datos":
    st.pyplot(fig1)
    st.pyplot(fig2) 
    

elif opt == "Segmentación de clientes":
    st.markdown("""
                En esta sección se realizó una segmentación de clientes utilizando el método **k-means**.
                
                El método *k-means* agrupa datos en *k* segmentos o "clusters" al minimizar la distancia entre cada punto de datos y el centro del grupo al que pertenece. 

                Matemáticamente, el objetivo de *k-means* es minimizar la suma de las distancias cuadradas entre cada punto $x_i$ y el centroide $\mu_j$ del grupo al que pertenece, según la fórmula:
                """)
    
    st.latex(r'''
                \text{min} \sum_{i=1}^n \sum_{j=1}^k || x_i - \mu_j ||^2
              ''')

    st.markdown("""
                Aquí:
                - $x_i$ es cada punto de datos.
                - $\mu_j$ es el centroide del *j*-ésimo grupo.
                - $|| x_i - \mu_j ||^2$ representa la distancia cuadrada entre el punto y su centroide.

                En este análisis, utilizamos *k = 4* para identificar cuatro tipos distintos de clientes. Las variables utilizadas para esta segmentación fueron: **Nº Noches**, **Nº Pasajeros**, **Precio por Noche**, **Mascota**, y **Pago en USD**. Estas variables permiten definir patrones específicos, o "huellas de identidad", para cada tipo de cliente.
                """)

    st.pyplot(fig3)
    st.write("### Para ver las diferencias entre cada tipo de cliente, se calculó el promedio de cada variable.")
    st.pyplot(fig4)
    
    st.markdown("""
                Para la fecha de entrega del informe y datos utilizados, se utilizó k=4, dando a cuenta cuatro tipo de clientes de la empresa.
                A saber, un perfil de gente que paga en USD que tiene a pagar un precio promedio por noche mayor (Cliente 4) o
                un tipo de cliente que paga en moneda CLP que tiende a ir en grandes grupos y por una estadía mayor (Cliente 2).
                Notandose además que para clientes que pagan poco por noche (Cliente 1 y 3), tienen estadias más cortas,
                mientras que aquello que pagan más noches, también pagan un precio x noche más elevado.
                """)

else:
    st.markdown("""
                En esta sección, se realizó un análisis mensual de la variable **Precio por Noche** para estudiar su comportamiento a lo largo del tiempo.

                Para modelar esta serie temporal, se utilizó un modelo **SARIMA** (Seasonal Autoregressive Integrated Moving Average). Este modelo es adecuado para datos que muestran patrones estacionales y se define mediante los parámetros $(p, d, q) \times (P, D, Q, s)$, donde:

                - **$p$**: Orden del componente autorregresivo.
                - **$d$**: Número de diferenciaciones necesarias para hacer la serie estacionaria.
                - **$q$**: Orden del componente de media móvil.
                - **$P$, $D$¡, $Q$**: Versiones estacionales de los parámetros anteriores.
                - **$s$**: Periodo estacional (por ejemplo, 12 para datos mensuales).

                La estructura del modelo SARIMA es:
                """)
    
    st.latex(r'''
                SARIMA(p, d, q) \times (P, D, Q, s)
              ''')

    st.markdown("""
                Además, para capturar la evolución de la **Precio por Noche** a través del tiempo, se utilizó el método **rolling window**. Este método realiza el ajuste del modelo en una ventana de tiempo móvil, recalculando los parámetros en cada paso para obtener predicciones actualizadas.
                """)

    st.pyplot(fig5)
    
    st.markdown("""
                De lo observado, se puede notar una fuente estacionalidad cada año respecto a los precios.
                """)


    


#col1, col2 = st.columns([2, 1])
#col1, col2 = st.columns([1,2]) 
#col3 = st.container()  
#with col1:
#    st.write("### Clientes")
#    st.pyplot(fig1)
#    st.pyplot(fig2)

#with col2:
#    st.write("### Datos")
#    st.pyplot(fig3) 

#with col3:
#   st.write("### Serie")
#   st.pyplot(fig4) 





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





