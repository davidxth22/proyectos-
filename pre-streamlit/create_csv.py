import pandas as pd

# Cargamos los datos
df = pd.read_excel('pre-streamlit/datos.xlsx')

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Eliminamos información sensible
df.drop(columns=['Nombre Completo', 'Teléfono'], inplace=True)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Crear la columna "Fecha" a partir de "Fecha Entrada" y extraer el día
df['Fecha'] = pd.to_datetime(df['Fecha Entrada'], errors='coerce')  # Convierte a tipo datetime y maneja errores

# Crear la columna "Periodo" en el formato "Año-Mes"
df['Periodo'] = df['Fecha'].dt.strftime('%Y-%m')  # '%Y' para el año y '%m' para el mes con ceros a la izquierda

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Crea columna de tipo de moneda: Asigna 1 si paga en USD y 0 si paga en CLP
df['Pago en USD'] = df['Precio USD'].notna().astype(int)

# Elimina los precios de estas monedas
df.drop(columns=['Precio USD', 'Precio CLP'], inplace=True)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Asignamos 1 si hubo servicio de limpieza "Clean Up" y 0 en caso contrario.
df['Clean Up'] = df['Clean Up'].apply(lambda x: 0 if x == 0 else 1).astype(int)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# Reorganiza las columnas, moviendo "Fecha" y "Periodo" a las primeras posiciones
df = df[['Fecha', 'Periodo', 'Año', 'Mes'] + [col for col in df.columns if col not in ['Fecha', 'Periodo', 'Año', 'Mes', 'Precio [$CLP] Neto', 'Pago en USD']] + ['Pago en USD', 'Precio [$CLP] Neto']] 

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Guardamos las observaciones con NaN en un archivo CSV antes de eliminarlas
df_nan = df[df.isna().any(axis=1)]

# Eliminamos las observaciones problemáticas con NaN solo después de aplicar todas las transformaciones
df.dropna(inplace=True)

# Guardamos el archivo solo si hay observaciones con NaN

df_nan.to_csv('obs_NaN.csv', index=False)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Guardamos número de pasajeros como entero
df['N° Pasajeros'] = df['N° Pasajeros'].astype(int)

# Guardamos número de noches como entero
df['N° Noches'] = df['N° Noches'].astype(int)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Arreglamos problema con presencia de Mascotas
df['Mascota'] = df['Mascota'].apply(lambda x: 0 if x == "No" else 1).astype(int)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Centramos la atención solo en el mes
df.drop(columns=['Fecha Entrada', 'Fecha Salida'], inplace=True)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Guardamos los datos finales en un archivo CSV
df.to_csv('pre-streamlit/datos.csv', index=False)

print("Los archivos 'datos.csv' y 'obs_NaN.csv' fueron creados/actualizados exitosamente.")
