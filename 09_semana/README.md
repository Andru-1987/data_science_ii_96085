# Guía Avanzada de Análisis Multivariado: De PCA a SVM y PLSR

Esta guía está diseñada para estructurar el conocimiento de manera progresiva, orientada a data engineers, data scientists y desarrolladores de IA que ya poseen bases estadísticas y de programación. El objetivo es profundizar en la teoría matemática, la implementación práctica y la toma de decisiones arquitectónicas al modelar datos multidimensionales.

---

## 1. Fundamentos del Análisis Multivariado

El [análisis multivariado](https://www.datacamp.com/es/tutorial/principal-component-analysis-in-python) estudia simultáneamente múltiples variables para identificar patrones, relaciones y estructuras complejas. A diferencia del análisis bivariado, permite modelar interacciones capturando la variabilidad conjunta.

**Perspectiva de Ingeniería (Álgebra Lineal):**
Los datos se representan como una matriz $X$ de dimensión $n \times p$ ($n$ observaciones, $p$ variables). Las técnicas operan sobre esta matriz mediante transformaciones lineales, descomposiciones (SVD, eigen-descomposición) y optimización de funciones objetivo.

### Objetivos y Aplicaciones Críticas

1. **Reducción de Dimensionalidad y Optimización:** Reducir miles de *features* a componentes principales minimiza el ruido y acelera el entrenamiento de modelos (ej. 10x de mejora en tiempos de cómputo en pipelines de ML).
2. **Feature Engineering en IA:** Preprocesamiento para redes neuronales, reducción de ruido en sensores y extracción de características (ej. eigenfaces en visión por computadora).
3. **Validación de Hipótesis:** Aplicación de MANOVA y regresión múltiple con validación cruzada para inferencia rigurosa.

**Consideraciones de Diseño:**

* **Ventaja:** Captura interacciones invisibles en 2D y mejora la generalización eliminando multicolinealidad.
* **Desventaja:** Alta sensibilidad a outliers y pérdida de interpretabilidad directa causal.
* **Regla de oro:** La estandarización previa (ej. `StandardScaler`) es obligatoria; de lo contrario, las variables con mayor magnitud dominarán los cálculos basados en varianza o distancia.

---

## 2. Exploración y Reducción: Análisis de Componentes Principales (PCA)

PCA es el pilar de la reducción de dimensionalidad no supervisada. Busca proyecciones ortogonales que maximicen la varianza de los datos.

### Implementación y Análisis (Ejemplo: Dataset Iris)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carga y Estandarización
iris = load_iris()
X, y, feature_names = iris.data, iris.target, iris.feature_names
X_scaled = StandardScaler().fit_transform(X) # CRÍTICO

# 2. Descomposición Matemática (Under the hood)
cov_matrix = np.cov(X_scaled.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
print(f"Varianza por eigenvalor: {np.round(eigen_vals / sum(eigen_vals), 3)}")

# 3. Implementación con scikit-learn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Varianza acumulada (2 PCs): {sum(pca.explained_variance_ratio_):.2%}")

# 4. Biplot: Interpretación de Loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=feature_names)

fig, ax = plt.subplots(figsize=(8, 6))
for target in np.unique(y):
    mask = y == target
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Clase {target}', alpha=0.6)

for i, feature in enumerate(feature_names):
    ax.arrow(0, 0, loadings['PC1'][i]*2, loadings['PC2'][i]*2, color='red', alpha=0.8, width=0.02)
    ax.text(loadings['PC1'][i]*2.1, loadings['PC2'][i]*2.1, feature, color='red')

ax.set_title('Biplot PCA - Dataset Iris')
ax.legend(); plt.show()

```

**Insights Arquitectónicos:**

* **Loadings:** Permiten trazar la relación inversa. Si un componente es vital para el modelo, los loadings revelan qué *features* originales alimentan esa decisión.
* **Monitoreo en Producción:** Evaluar la varianza explicada por los componentes en inferencia. Una caída drástica indica *data drift*.

---

## 3. Regresión Multivariada Avanzada: De PCR a PLSR

Mientras que PCA maximiza la varianza interna de $X$ (ignorando la variable objetivo $Y$), **PLSR (Partial Least Squares Regression)** proyecta tanto $X$ como $Y$ a un nuevo espacio, maximizando la covarianza entre ellos.

### PLSR: Fundamento Matemático

El modelo asume que $Y = XB + E$. PLSR extrae componentes iterativamente:

1. Calcula pesos $w_h$ que maximizan la covarianza: $w_h = \arg\max_{w} \text{cov}(Xw, Y)$
2. Calcula *scores* ($t_h = Xw_h$) y *loadings* para residualizar (deflactar) las matrices $X$ e $Y$ en cada paso.

### Comparativa Estratégica: PCR vs PLSR

| Aspecto | PCR (Principal Component Regression) | PLSR (Partial Least Squares) |
| --- | --- | --- |
| **Objetivo** | Maximizar varianza de $X$ | Maximizar covarianza entre $X$ e $Y$ |
| **Componentes** | Requiere más componentes | Más eficiente (menos componentes) |
| **Rendimiento** | Riesgo de perder información útil para Y | Superior con alta correlación $X$-$Y$ |
| **Caso Ideal** | Multicolinealidad sin respuesta definida | Quimiometría, espectroscopía, variables altamente correlacionadas |

### Implementación de PLSR con Optimización

Ejemplo de predicción de caballos de fuerza (`hp`) usando variables correlacionadas de vehículos.

```python
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# Asumiendo X e y cargados y estandarizados (X_scaled, y_scaled)
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mse_scores = []
max_comp = X.shape[1]

# Búsqueda del número óptimo de componentes latentes
for i in range(1, max_comp + 1):
    pls = PLSRegression(n_components=i)
    y_cv = cross_val_predict(pls, X_scaled, y_scaled, cv=cv)
    mse_scores.append(mean_squared_error(y_scaled, y_cv))

n_opt = np.argmin(mse_scores) + 1
pls_final = PLSRegression(n_components=n_opt).fit(X_scaled, y_scaled)

print(f"Componentes óptimos encontrados: {n_opt}")
# Extracción de la importancia (VIP puede programarse en base a pls_final.x_weights_)

```

---

## 4. Modelado No Lineal: SVM Multivariado

Cuando las relaciones superan la linealidad de PLSR o PCA, las **Máquinas de Vectores de Soporte (SVM)** entran en juego, mapeando datos a espacios de mayor dimensión mediante el *Kernel Trick*.

### Formulación Dual y Kernels

El objetivo es maximizar el margen resolviendo:


$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j K(x_i, x_j)$$


Donde $K(x_i, x_j)$ representa transformaciones como el **Kernel Lineal** (textos de alta dimensión), **Polinómico** o **RBF/Gaussiano** (por defecto para fronteras complejas).

### Estrategias Multivariadas (Clasificación Multiclase)

SVM es inherentemente binario. Para $K > 2$ clases, el motor subyacente adapta la arquitectura:

1. **One-vs-One (OvO):** Entrena $K(K-1)/2$ modelos. Menos sesgo de clases, estándar en `scikit-learn` (`SVC`).
2. **One-vs-Rest (OvR):** Entrena $K$ modelos. Computacionalmente más ligero ($O(K)$) pero sufre de desbalance intrínseco.

### Implementación: Multi-Output Regression SVR

Cuando se predicen múltiples variables continuas simultáneamente:

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# Dataset con 3 variables objetivo
X, y = make_regression(n_samples=500, n_features=10, n_targets=3, noise=0.1)

# El MultiOutputRegressor entrena un SVR por cada target
multi_svr = MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma='scale'))
multi_svr.fit(X, y)

# Arquitectura resultante:
print(f"Modelos subyacentes instanciados: {len(multi_svr.estimators_)}")

```

*Tip de Arquitectura:* `MultiOutputRegressor` asume independencia entre los outputs. Si las variables respuesta están correlacionadas, arquitecturas como Redes Neuronales Multi-Task o PLSR son matemáticamente superiores.

---

## 5. Comparativa Decisional: PLSR vs SVM

| Característica | PLSR | SVM Multivariado |
| --- | --- | --- |
| **Naturaleza** | Lineal (espacio latente) | Lineal o No lineal (Kernels) |
| **Output (Y) Múltiple** | Nativo, aprovecha correlaciones de Y | Requiere wrappers (ignora correlación entre Y's) |
| **Interpretabilidad** | Alta (Loadings, VIP Scores) | Baja (Caja negra en Kernels no lineales) |
| **Escalabilidad** | Excelente ($O(n \cdot p \cdot k)$) | Pobre en inferencia masiva ($O(n^2)$ a $O(n^3)$) |

---

## 6. Revisión de Pares y Comunicación Técnica

Construir el pipeline es solo la mitad del trabajo; validar y comunicar los hallazgos define el impacto en el negocio.

### Checklists para Code Reviews (Data/ML Engineering)

* **Data Leakage:** ¿Se aplicó `fit_transform` del `StandardScaler` y de `PCA` a todo el dataset antes del split? (Debería ser ajustado **solo** en train mediante un `Pipeline`).
* **Optimismo de Validación:** ¿Se usó *Nested Cross-Validation* para el ajuste de hiperparámetros (C, gamma en SVM) y la selección de componentes (PLSR)?

### Estrategias de Presentación Eficiente

1. **Tablas vs Gráficos:** Use tablas para menos de 20 puntos de datos numéricos críticos. Para más, utilice visualizaciones (ej. Heatmaps de correlación, Biplots).
2. **Estándar de Reporte (APA 7):** Evite ambigüedades. *Mal:* "P-value es bajo". *Bien:* $t(48) = 2.34, p = .023, 95\% \text{ CI } [0.12, 0.89]$.
3. **Narrativa de Negocio:** No diga "El kernel RBF convergió con $C=10$ capturando 90% de varianza". Diga "El algoritmo segmentó correctamente a los clientes premium basándose en interacciones complejas de uso, logrando un 90% de precisión esperada en producción."

---

## 7. Glosario Técnico Clave

* **Varianza Explicada:** Proporción de la señal original retenida por un hiperplano o componente.
* **Deflactar (Residualizar):** En PLSR, el proceso de sustraer la varianza ya explicada por un componente antes de buscar el siguiente.
* **Kernel Trick:** Proyectar implícitamente datos a un espacio de mayor dimensión calculando solo el producto punto entre vectores.
* **Hiperparámetro $C$ (SVM):** Controla la regularización. Valores altos penalizan fuertemente los errores (riesgo de *overfitting*), valores bajos priorizan márgenes suaves (*underfitting*).
* **VIP (Variable Importance in Projection):** Métrica en PLSR que consolida los pesos de una variable a través de todos los componentes latentes.
