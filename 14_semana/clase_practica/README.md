### 1. Configuración del Entorno y Estrategia de Validación

* **Reproducibilidad:** Se fija una semilla aleatoria uniforme en todos los estimadores y particiones para garantizar la comparabilidad exacta entre experimentos.
* **Esquema de Validación Cruzada:** Se implementa una validación cruzada estratificada de 5 pliegues ($K=5$). Esto asegura que cada pliegue mantenga la misma proporción de la clase minoritaria (inactivos) que el conjunto de datos completo, evitando sesgos por distribución.
* **Métrica Objetivo Principal (Métrica de Tuning):** Se utiliza el **PR-AUC** (*Precision-Recall Area Under Curve* o `average_precision`). A diferencia del ROC-AUC o del Accuracy, el PR-AUC se enfoca en la capacidad del modelo para ordenar la clase positiva/minoritaria en datasets con desbalance de clases, sirviendo como métrica de ranking sin depender de un umbral prefijado.

---

### 2. Definición de la Población y el Target Binario

* **Filtrado de Población:** Se descartan publicaciones con un mínimo de noches superior o igual a 30 (`minimum_nights >= 30`). Negociablemente corresponden a alquileres de larga estancia cuyo comportamiento habitual difiere del turismo de corta estancia (donde pocas reseñas son esperables y no implican inactividad).
* **Definición del Target ($y$):** Se define un target binario estricto: $1$ para avisos **inactivos** (cero reseñas en los últimos 12 meses) y $0$ para avisos **activos**.
* **Preprocesamiento Estático:** Se calcula la transformación logarítmica de la variable precio ($\log(1 + \text{price})$). Al ser una transformación fila a fila independiente, no requiere estadísticos globales del conjunto y no genera fuga de datos (*data leakage*).

---

### 3. Aislamiento y Prevención de Fuga de Datos (*Data Leakage*)

* **Eliminación de Variables Delatoras:** Se descartan del conjunto de características ($X$) todas las variables asociadas a métricas históricas de reseñas (`number_of_reviews`, `reviews_per_month`, `last_review`). Al estar matemáticamente o temporalmente vinculadas con el target, incorporarlas inflaría artificialmente el rendimiento del modelo.
* **Evaluación de Variables Operativas:** Se conserva la disponibilidad en días (`availability_365`), considerando que aunque es un indicador operativo del estado del anuncio, no viola la causalidad temporal y aporta capacidad predictiva.

---

### 4. Partición del Dataset y Diseño del Pipeline Unificado

* **Split Estratificado:** Se divide el conjunto total en dos bloques aislados: Desarrollo/Entrenamiento (**80%**, $X_{\text{dev}}$) y Test Final (**20%**, $X_{\text{test}}$), aplicando estratificación según el target. El conjunto de test se reserva de forma ciega hasta la evaluación final.
* **Pipeline de Preprocesamiento (`ColumnTransformer`):**
* **Variables Numéricas:** Se procesan mediante imputación mediana para valores faltantes, seguida de un escalado estándar (`StandardScaler`) o una discretización por cuantiles (`KBinsDiscretizer`). Esta última permite a los modelos lineales capturar efectos no lineales por tramos.
* **Variables Categóricas:** Se codifican mediante *One-Hot Encoding*, configurando la tolerancia a categorías no vistas (`handle_unknown='ignore'`) para garantizar la robustez en inferencia.


* **Garantía Anti-Fuga:** Las transformaciones (mEDIANAS, medias y desviaciones estándar) se ajustan únicamente sobre el pliegue de entrenamiento dentro de cada ciclo de validación cruzada.

---

### 5. Benchmark de Modelos en Validación Cruzada

Se evalúan simultáneamente cinco arquitecturas sobre los mismos pliegues de validación cruzada:

1. **Dummy Classifier:** Modelo de referencia nulo que predice según la prevalencia de la clase. Establece el piso teórico de PR-AUC (igual a la tasa de inactivos).
2. **Regresión Logística:** Baseline lineal paramétrico.
3. **Regresión Logística con Discretización (Bins):** Extensión del modelo lineal con capacidad para modelar relaciones no monótonas por tramos.
4. **Random Forest:** Ensamble no lineal mediante *bagging* de árboles de decisión no restringidos.
5. **HistGradientBoosting (HGB):** Ensamble por *boosting* basado en histogramas, eficiente para datos tabulares complejos e interacciones no lineales.

Se obtienen y tabulan las métricas promedio de $ROC-AUC$, $PR-AUC$ y $F_1$ (a umbral por defecto 0.5), realizando comparaciones pareadas fold a fold entre competidores.

---

### 6. Cuantificación del Aporte de Variables Clave

* Se realiza un análisis de ablación removiendo temporalmente la variable `availability_365` en todos los modelos.
* Se mide el delta ($\Delta$) de PR-AUC obtenido para diferenciar si la señal predictiva proviene de la estructura general del aviso (ubicación, precio, tipo de habitación) o si depende críticamente del comportamiento del calendario del anfitrión.

---

### 7. Selección del Modelo Ganador (Principio de Parsimonia)

* **Evaluación de Linealidad:** Se analiza la brecha entre la Regresión Logística, la versión con *bins* y los modelos de ensambles (*RF* / *HGB*).
* **Regla de 1 Desvío Estándar:** Para evitar seleccionar modelos sobre-complejos por ganancias marginales de rendimiento, se elige el estimador estructuralmente **más simple** cuyo PR-AUC medio se mantenga dentro del margen de un desvío estándar respecto al modelo con mayor puntaje absoluto en CV.

---

### 8. Diagnóstico de Curvas de Complejidad (Sesgo vs. Varianza)

* Se evalúa el comportamiento del modelo frente a variaciones en sus parámetros de complejidad (ej. `max_depth` en Random Forest o `max_leaf_nodes` en HGB) aplicando `validation_curve` sobre $X_{\text{dev}}$.
* Se analizan tres regiones en el gráfico de entrenamiento vs. validación:
* **Bajo ajuste (Underfitting):** Puntajes bajos tanto en entrenamiento como en validación.
* **Punto de Generalización Óptima:** Máximo rendimiento alcanzado en el conjunto de validación.
* **Sobreajuste (Overfitting):** Divergencia donde el puntaje de entrenamiento tiende a 1.0 mientras que el de validación se estanca o decae (incremento de la brecha de memorización).



---

### 9. Optimización de Hiperparámetros (Tuning Controlado)

* **Presupuesto Fijo:** Se establece una cuota máxima de iteraciones ($N_{\text{trials}} = 10$) bajo exactamente los mismos folds para garantizar una comparación transparente.
* **Estrategias Comparadas:**
1. **RandomizedSearchCV:** Muestreo aleatorio dentro del espacio categórico/discreto prefijado.
2. **Optuna (TPESampler):** Búsqueda bayesiana mediante *Tree-structured Parzen Estimator*, que utiliza la historia de evaluaciones para guiar el muestreo hacia las regiones de mayor rendimiento.


* **Resultado:** Se retiene la mejor combinación de hiperparámetros según la métrica PR-AUC en CV.

---

### 10. Optimización del Umbral de Decisión Operativo (Out-of-Fold)

* **Predicciones Fuera del Pliegue (*Out-of-Fold*):** Se generan las probabilidades predichas sobre $X_{\text{dev}}$ mediante `cross_val_predict`, asegurando que el umbral no se optimice sobre datos vistos durante el entrenamiento de los submodelos.
* **Criterio de Negocio:** Se ajusta el umbral $t \in [0, 1]$ utilizando la curva Precisión-Recall:
* **Objetivo:** Maximizar el *Recall* (capturar la mayor cantidad de inactivos reales) garantizando una **Precisión mínima ($\ge 60\%$)** (asegurar que al menos 6 de cada 10 avisos auditados sean efectivamente inactivos).
* Si la restricción del 60% no es alcanzable debido a la separación de los datos, la regla degrada gradualmente la exigencia a metas realizables (50%, 40% o 30%).



---

### 11. Evaluación Unblinded en Test Set y Diagnóstico Final

* **Evaluación Única:** Se aplica el modelo optimizado y reentrenado sobre la totalidad de $X_{\text{dev}}$ para precedir sobre el conjunto ciega $X_{\text{test}}$.
* **Métricas de Diagnóstico:**
* **Métricas Globales de Ranking:** Reporte final de $ROC-AUC$ y $PR-AUC$ comparados contra la Regresión Logística inicial.
* **Matriz de Confusión y Reporte de Clasificación:** Evaluación a nivel operativo del número de Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos aplicando el umbral de negocio seleccionado.
* **Validación de Consistencia:** Comparación del porcentaje de Precisión y Recall esperados en Validación Cruzada versus los efectivamente observados en el Test Set para descartar el sobreajuste del umbral o la fuga de datos.



---

### 12. Caso de Uso e Inferencia sobre Nuevas Observaciones

* **Inferencia Directa:** El pipeline entrenado procesa registros crudos de nuevos anuncios (transformando el precio e imputando/escalando variables en tiempo de ejecución).
* **Salida Operativa:** Se devuelven dos valores por cada anuncio:
1. `p_inactivo`: Probabilidad continua estimada de estar inactivo.
2. `auditar`: Indicador booleano (`True`/`False`) resultante de comparar la probabilidad contra el umbral optimizado, dictaminando qué publicaciones deben derivarse al equipo de auditoría de oferta.
