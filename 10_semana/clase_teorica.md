**Punto clave:** Dado que la variable objetivo (`Sales`) es un valor numérico continuo, estamos frente a un problema de **Regresión**, no de Clasificación. Por lo tanto, los algoritmos utilizados cambian (ej. de `RandomForestClassifier` a `RandomForestRegressor`) y la métrica de evaluación pasa a ser el error (ej. MSE) o el $R^2$.

Además, como este dataset solo contiene variables numéricas (`TV`, `Radio`, `Newspaper`), he incluido un paso donde **creamos artificialmente una variable categórica** (discretización) para que puedas demostrar las técnicas de codificación .

### 1. Carga de Datos y Preprocesamiento (Creación de Categorías)

Este bloque demuestra cómo cargar el archivo y cómo aplicar **Feature Engineering** para transformar una variable continua (inversión en TV) en una variable categórica (Baja, Media, Alta), lo cual es útil para segmentación comercial.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# 1. Cargar el dataset (asumiendo que se guarda como 'advertising.csv')
df = pd.read_csv('advertising.csv', index_col=0)

# Separar features (X) y target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print("Vista inicial del dataset:")
print(X.head(), "\n")

# 2. Discretización (Binning): Convertir 'TV' en categórica para demostración
# Divide la inversión de TV en 3 grupos: Baja, Media, Alta
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
df['TV_Categoria'] = kbins.fit_transform(df[['TV']])

print("Dataset con nueva característica categórica (0=Baja, 1=Media, 2=Alta):")
print(df[['TV', 'TV_Categoria']].head(), "\n")

# 3. One-Hot Encoding sobre la nueva categoría
ohe = OneHotEncoder(sparse_output=False, drop='first')
tv_ohe = ohe.fit_transform(df[['TV_Categoria']])

# Convertir a DataFrame para mostrar a los alumnos
ohe_df = pd.DataFrame(tv_ohe, columns=['TV_Media', 'TV_Alta'])
print("Transformación One-Hot Encoding (evitando multicolinealidad):")
print(ohe_df.head())

```

---

### 2. Ingeniería de Características (Interacciones)

En publicidad, la suma de TV y Radio suele tener un efecto sinérgico (el impacto combinado es mayor que la suma individual). Este bloque muestra cómo Scikit-Learn detecta estas interacciones matemáticamente.

```python
from sklearn.preprocessing import PolynomialFeatures

# Generar interacciones polinómicas (grado 2)
# interaction_only=True evita generar TV^2, Radio^2, solo genera TV*Radio
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interacciones = poly.fit_transform(X)

print("\nNuevas variables (Features) generadas:")
print(poly.get_feature_names_out(['TV', 'Radio', 'Newspaper']))

print("\nMatriz con la interacción sinérgica calculada (Primeras 2 filas):")
print(X_interacciones[:2])

```

---

### 3. Optimización de Hiperparámetros (Búsqueda de la Mejor Regresión)

Aquí demuestras cómo encontrar la mejor configuración para un modelo predictivo de ventas utilizando **GridSearchCV**. Al ser regresión, usamos `neg_mean_squared_error` (buscamos el error más cercano a cero) o `r2` (buscamos acercarnos a 1).

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Iniciar el modelo base
rf_model = RandomForestRegressor(random_state=42)

# Definir el espacio de búsqueda (Hiperparámetros a probar)
param_grid = {
    'n_estimators': [50, 100, 200], # Cantidad de árboles
    'max_depth': [None, 5, 10],     # Profundidad máxima
    'min_samples_split': [2, 5, 10] # Muestras mínimas para dividir un nodo
}

# Configurar GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5, # Validación cruzada de 5 pliegues
    scoring='r2', # Métrica para evaluación
    n_jobs=-1 # Usar todos los núcleos del procesador
)

# Entrenar evaluando todas las combinaciones
grid_search.fit(X, y)

print("\n--- Resultados de la Optimización ---")
print("Mejores Hiperparámetros encontrados:")
print(grid_search.best_params_)
print(f"Mejor Score R2 (Precisión de explicación de varianza): {grid_search.best_score_:.4f}")

```

---

### 4. Arquitectura de Producción: Pipeline Completo

Integracion todo el flujo. Estructura el código para producción, encapsulando la creación de interacciones, el escalado de variables numéricas y el modelo predictivo en un solo objeto para evitar el _Data Leakage_ (fuga de datos).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 1. Construir el Pipeline
# El flujo entra puro y se transforma secuencialmente
pipeline_ventas = Pipeline([
    ('interacciones', PolynomialFeatures(degree=2, include_bias=False)),
    ('escalador', StandardScaler()),
    ('modelo', RandomForestRegressor(random_state=42))
])

# 2. Espacio de búsqueda apuntando a los pasos del pipeline
# Nota para la clase: Se usa el prefijo 'modelo__' para indicar a qué paso pertenece el parámetro
param_dist_pipe = {
    'modelo__n_estimators': randint(100, 300),
    'modelo__max_depth': randint(5, 20)
}

# 3. Optimización eficiente con RandomizedSearchCV sobre el Pipeline
random_pipe = RandomizedSearchCV(
    estimator=pipeline_ventas,
    param_distributions=param_dist_pipe,
    n_iter=10, # Probar 10 combinaciones aleatorias
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

# 4. Entrenar todo el flujo
random_pipe.fit(X, y)

print("\n--- Resultados del Pipeline de Producción ---")
print("Mejor configuración del pipeline:")
print(random_pipe.best_params_)
print(f"Mejor Score R2 en validación cruzada: {random_pipe.best_score_:.4f}")

# Simulación de predicción con un dato nuevo:
nuevo_presupuesto = pd.DataFrame({'TV': [150], 'Radio': [40], 'Newspaper': [20]})
prediccion = random_pipe.predict(nuevo_presupuesto)
print(f"\nPredicción de ventas para el nuevo presupuesto: {prediccion[0]:.2f} unidades")

```
