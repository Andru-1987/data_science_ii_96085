"""
Análisis Exploratorio de Datos (EDA) en Python
Dataset: Cars Dataset (CooperUnion / Kaggle)
"""

# ==========================================
# 1. Importación de librerías requeridas
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración del estilo visual para los gráficos
sns.set(color_codes=True)


# ==========================================
# 2. Carga del conjunto de datos en un DataFrame
# ==========================================
# Se lee el archivo CSV con los datos de los vehículos
df = pd.read_csv("cars-dataset.csv")

# Mostrar las primeras 5 filas del dataset
print("--- Primeras 5 filas del dataset ---")
print(df.head(5))

# Mostrar las últimas 5 filas del dataset
print("\n--- Últimas 5 filas del dataset ---")
print(df.tail(5))


# ==========================================
# 3. Comprobación de tipos de datos
# ==========================================
# Verificar los tipos de datos de cada columna
print("\n--- Tipos de datos por columna ---")
print(df.dtypes)


# ==========================================
# 4. Eliminación de columnas irrelevantes
# ==========================================
# Se eliminan características que no se utilizarán en el análisis/modelado
columnas_a_eliminar = [
    'Engine Fuel Type', 
    'Market Category', 
    'Vehicle Style', 
    'Popularity', 
    'Number of Doors', 
    'Vehicle Size'
]
df = df.drop(columnas_a_eliminar, axis=1)

print("\n--- Dataset tras eliminar columnas irrelevantes ---")
print(df.head(5))


# ==========================================
# 5. Renombrar columnas para mayor legibilidad
# ==========================================
# Simplificación de los nombres de columnas
nombres_nuevos = {
    "Engine HP": "HP",
    "Engine Cylinders": "Cylinders",
    "Transmission Type": "Transmission",
    "Driven_Wheels": "Drive Mode",
    "highway MPG": "MPG-H",
    "city mpg": "MPG-C",
    "MSRP": "Price"
}
df = df.rename(columns=nombres_nuevos)

print("\n--- Dataset con columnas renombradas ---")
print(df.head(5))


# ==========================================
# 6. Detección y eliminación de duplicados
# ==========================================
# Dimensión original antes de remover duplicados
print(f"\nDimensión original: {df.shape}")

# Conteo y visualización de filas duplicadas
filas_duplicadas = df[df.duplicated()]
print(f"Número de filas duplicadas: {filas_duplicadas.shape[0]}")

# Conteo inicial de registros por columna
print("\n--- Conteo previo a eliminar duplicados ---")
print(df.count())

# Eliminación de registros duplicados
df = df.drop_duplicates()

print("\n--- Conteo posterior a eliminar duplicados ---")
print(df.count())
print(f"Nueva dimensión del DataFrame: {df.shape}")


# ==========================================
# 7. Manejo y eliminación de valores nulos (NaN)
# ==========================================
# Identificar cantidad de valores nulos por columna
print("\n--- Valores nulos por columna ---")
print(df.isnull().sum())

# Eliminación de filas con valores faltantes
df = df.dropna()

print("\n--- Conteo tras eliminar valores nulos ---")
print(df.count())
print(f"Valores nulos restantes:\n{df.isnull().sum()}")


# ==========================================
# 8. Detección y filtrado de Outliers (IQR)
# ==========================================
# Visualización previa con diagramas de caja (Boxplots)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Price'])
plt.title("Distribución de Precio (Price) con Outliers")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['HP'])
plt.title("Distribución de Caballos de Fuerza (HP) con Outliers")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Cylinders'])
plt.title("Distribución de Cilindros (Cylinders) con Outliers")
plt.show()

# Cálculo del Rango Intercuartílico (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("\n--- Rango Intercuartílico (IQR) por columna numérica ---")
print(IQR)

# Filtrado de valores atípicos utilizando el criterio 1.5 * IQR
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"\nDimensión del DataFrame tras remover outliers: {df.shape}")


# ==========================================
# 9. Visualizaciones y Análisis de Relaciones
# ==========================================

# 9.1 Histograma / Gráfico de barras: Top fabricantes
plt.figure(figsize=(10, 5))
df.Make.value_counts().nlargest(40).plot(kind='bar')
plt.title("Número de automóviles por marca (Top 40)")
plt.ylabel("Cantidad de automóviles")
plt.xlabel("Marca (Make)")
plt.tight_layout()
plt.show()

# 9.2 Mapa de calor (Heatmap) de correlación
plt.figure(figsize=(10, 5))
matriz_correlacion = df.corr()
sns.heatmap(matriz_correlacion, cmap="BrBG", annot=True, fmt=".2f")
plt.title("Matriz de Correlación entre Variables")
plt.tight_layout()
plt.show()

# 9.3 Gráfico de dispersión (Scatterplot): HP vs. Precio
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['HP'], df['Price'], alpha=0.5)
ax.set_title("Relación entre Caballos de Fuerza (HP) y Precio (Price)")
ax.set_xlabel("Caballos de Fuerza (HP)")
ax.set_ylabel("Precio (Price)")
plt.tight_layout()
plt.show()
