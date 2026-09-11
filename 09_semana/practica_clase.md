### Environment: Generación de Datasets

Ejecuta este bloque de código primero. Generará los entornos sintéticos (datasets) necesarios para aislar y estudiar el comportamiento de PCA, PLSR y SVM.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Dataset 1: Para PCA (5 dimensiones, 3 clases, con características redundantes)
X_pca_raw, y_pca = make_classification(
    n_samples=300, n_features=5, n_informative=3, n_redundant=2,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
scaler_pca = StandardScaler()
X_pca = scaler_pca.fit_transform(X_pca_raw)
df_pca = pd.DataFrame(X_pca, columns=['F1', 'F2', 'F3', 'F4', 'F5'])
df_pca['Class'] = y_pca.astype(str)

# Dataset 2: Para PLSR (Multicolinealidad extrema: 15 features, pero rango efectivo 3)
X_pls_raw, y_pls = make_regression(
    n_samples=200, n_features=15, n_informative=3, effective_rank=3,
    noise=0.5, random_state=42
)
scaler_pls_X = StandardScaler()
scaler_pls_y = StandardScaler()
X_pls = scaler_pls_X.fit_transform(X_pls_raw)
y_pls = scaler_pls_y.fit_transform(y_pls.reshape(-1, 1)).ravel()

# Dataset 3: Para SVM Multivariado (Fronteras no lineales en 2D)
X_svm_raw, y_svm = make_moons(n_samples=300, noise=0.15, random_state=42)
scaler_svm = StandardScaler()
X_svm = scaler_svm.fit_transform(X_svm_raw)

```

---

### Ejercicio 1: Análisis de Componentes Principales (PCA) en 3D

* **Enunciado**
En datasets de alta dimensionalidad, las variables suelen estar correlacionadas, introduciendo redundancia y ruido. Se requiere proyectar un espacio de 5 dimensiones a 3 componentes principales ortogonales, analizando la estructura latente de las clases.
* **Ejercicio**
Aplica PCA sobre el `Dataset 1`. Extrae los 3 primeros componentes principales y calcula su varianza explicada. Utiliza Plotly Express para generar un *scatter plot* en 3D que mapee estos componentes, coloreando por clase e indicando el porcentaje de varianza en cada eje.
* **Codigo**

```python
import plotly.express as px
from sklearn.decomposition import PCA

# 1. Aplicar PCA
pca = PCA(n_components=3)
X_pca_proj = pca.fit_transform(X_pca)
var_explicada = pca.explained_variance_ratio_ * 100

# 2. Preparar dataframe para Plotly
df_pca_proj = pd.DataFrame(
    X_pca_proj, 
    columns=['PC1', 'PC2', 'PC3']
)
df_pca_proj['Class'] = df_pca['Class']

# 3. Visualización 3D interactiva
fig_pca = px.scatter_3d(
    df_pca_proj, x='PC1', y='PC2', z='PC3',
    color='Class',
    title="Proyección PCA 3D - Espacio Latente",
    labels={
        'PC1': f"PC1 ({var_explicada[0]:.1f}%)",
        'PC2': f"PC2 ({var_explicada[1]:.1f}%)",
        'PC3': f"PC3 ({var_explicada[2]:.1f}%)"
    },
    color_discrete_sequence=px.colors.qualitative.Vivid
)

fig_pca.update_traces(marker=dict(size=4, opacity=0.8))
fig_pca.show()

```

* **Justificacion**
Al rotar el gráfico 3D generado por Plotly, se puede validar visualmente si el hiperplano conformado por los 3 primeros componentes (que en este caso capturarán la mayoría de la varianza útil dado que `n_informative=3` en la generación) es suficiente para separar las clases. Matemáticamente, PCA realizó una descomposición de la matriz de covarianza de `X_pca`; los ejes del gráfico (PC1, PC2, PC3) representan los autovectores con los autovalores más altos.

---

### Ejercicio 2: Partial Least Squares Regression (PLSR) vs Multicolinealidad

* **Enunciado**
Se dispone de un set de datos (`Dataset 2`) con 15 variables predictoras altamente correlacionadas (rango efectivo = 3). Aplicar Regresión Lineal Múltiple directa causaría inestabilidad en los coeficientes por multicolinealidad.
* **Ejercicio**
Entrena un modelo PLSR optimizando a 3 componentes latentes. Evalúa la capacidad predictiva ploteando los valores Reales vs Predichos. Luego, extrae e inspecciona visualmente los coeficientes de regresión resultantes para identificar las variables de mayor impacto.
* **Codigo**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

# 1. Entrenar modelo PLSR
pls = PLSRegression(n_components=3)
pls.fit(X_pls, y_pls)
y_pred_pls = pls.predict(X_pls).ravel()
r2 = r2_score(y_pls, y_pred_pls)

# 2. Configurar subplots (Real vs Predicho y Coeficientes)
fig_pls = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(f"Real vs Predicho (R2: {r2:.3f})", "Coeficientes de Regresión PLSR")
)

# Plot 1: Scatter Real vs Predicho
fig_pls.add_trace(
    go.Scatter(x=y_pls, y=y_pred_pls, mode='markers', 
               marker=dict(color='blue', opacity=0.6), name="Observaciones"),
    row=1, col=1
)
# Línea ideal (y = x)
fig_pls.add_trace(
    go.Scatter(x=[y_pls.min(), y_pls.max()], y=[y_pls.min(), y_pls.max()],
               mode='lines', line=dict(color='red', dash='dash'), name="Ajuste Ideal"),
    row=1, col=1
)

# Plot 2: Bar chart de Coeficientes
coeficientes = pls.coef_.ravel()
nombres_features = [f"F{i+1}" for i in range(len(coeficientes))]

fig_pls.add_trace(
    go.Bar(x=nombres_features, y=coeficientes, marker_color='teal', name="Impacto Variable"),
    row=1, col=2
)

fig_pls.update_layout(title_text="Análisis PLSR: Predicción e Importancia de Variables", showlegend=False)
fig_pls.update_xaxes(title_text="Valor Real", row=1, col=1)
fig_pls.update_yaxes(title_text="Valor Predicho", row=1, col=1)
fig_pls.show()

```

* **Justificacion**
A diferencia de PCA, PLSR proyecta simultáneamente $X$ e $y$ hacia un nuevo espacio maximizando su covarianza. El primer gráfico demuestra la bondad de ajuste libre de sobreajuste crítico. El segundo gráfico es fundamental en ciencia de datos: muestra los coeficientes finales ($\beta$) proyectados de vuelta al espacio original de 15 variables. PLSR maneja la multicolinealidad distribuyendo los pesos lógicamente sin que las matrices se vuelvan singulares, algo que destruiría una regresión OLS estándar.

---

### Ejercicio 3: Máquinas de Vectores de Soporte (SVM) y Fronteras No Lineales

* **Enunciado**
En problemas de clasificación multivariada donde las clases no son linealmente separables, los métodos de proyección lineal fallan. Se requiere mapear los datos a un espacio de mayor dimensión para encontrar un hiperplano de separación óptimo.
* **Ejercicio**
Entrena un clasificador SVM utilizando un Kernel Radial Basis Function (RBF) sobre el `Dataset 3` (lunas intercaladas). Genera una malla (grid) espacial, calcula la función de decisión del modelo sobre dicha malla y utiliza un gráfico de contorno de Plotly para visualizar la frontera de decisión continua, superponiendo los datos reales.
* **Codigo**

```python
from sklearn.svm import SVC

# 1. Entrenar modelo SVM con Kernel RBF
# C controla la regularización; gamma controla el radio de influencia (kernel)
svm_model = SVC(kernel='rbf', C=10, gamma=1.0)
svm_model.fit(X_svm, y_svm)

# 2. Crear meshgrid para la frontera de decisión
h = 0.02  # Tamaño del paso en la malla
x_min, x_max = X_svm[:, 0].min() - 0.5, X_svm[:, 0].max() + 0.5
y_min, y_max = X_svm[:, 1].min() - 0.5, X_svm[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 3. Predecir la función de decisión (distancia al hiperplano)
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Gráfico interactivo Plotly (Contour + Scatter)
fig_svm = go.Figure()

# Capa 1: Contorno continuo de la función de decisión
fig_svm.add_trace(go.Contour(
    x=np.arange(x_min, x_max, h),
    y=np.arange(y_min, y_max, h),
    z=Z,
    colorscale='RdBu',
    opacity=0.5,
    showscale=True,
    contours=dict(start=-1.5, end=1.5, size=0.5)
))

# Capa 2: Puntos de datos originales
fig_svm.add_trace(go.Scatter(
    x=X_svm[:, 0], y=X_svm[:, 1],
    mode='markers',
    marker=dict(
        color=y_svm,
        colorscale='RdBu',
        line=dict(color='black', width=1),
        size=8
    ),
    name='Datos'
))

# Capa 3: Marcar los vectores de soporte
support_vectors = svm_model.support_vectors_
fig_svm.add_trace(go.Scatter(
    x=support_vectors[:, 0], y=support_vectors[:, 1],
    mode='markers',
    marker=dict(symbol='circle-open', size=12, line=dict(width=2, color='green')),
    name='Vectores de Soporte'
))

fig_svm.update_layout(
    title="Frontera de Decisión No Lineal - SVM (Kernel RBF)",
    xaxis_title="Variable X1 estandarizada",
    yaxis_title="Variable X2 estandarizada"
)
fig_svm.show()

```

* **Justificacion**
El Kernel Trick permite a SVM resolver separaciones no lineales complejas computando los productos punto en un espacio dimensional superior implícito. El uso de `decision_function` mapeado sobre un `go.Contour` de Plotly revela la topografía exacta de este hiperplano: la línea donde $Z=0$ es la frontera de decisión exacta. Los círculos verdes resaltan los vectores de soporte; matemáticamente, son las únicas observaciones que dictan la posición de la frontera de decisión, demostrando la eficiencia estructural del algoritmo SVM frente a otros modelos predictivos.
