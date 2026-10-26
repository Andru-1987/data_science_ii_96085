## Estructura de la Clase Práctica: Predicción de Precios de Alquiler (Buenos Aires Airbnb)

El dataset de alquileres temporales de Airbnb en Buenos Aires (`buenos_aires.csv`, 29.753 registros) es ideal para este propósito. Nos permite abordar un problema real de **predicción de precios (`price`)** mediante regresión, donde conviven variables categóricas, numéricas, valores faltantes y sesgos de distribución (outliers).

---

### Preparación del entorno y dataset

* **Carga y limpieza básica:** se descartan los registros sin `price`, ya que el target debe estar disponible.
* **Filtrado de outliers extremos:** se conservan precios entre 10.000 y 1.000.000 ARS.
* **Features:** `neighbourhood`, `room_type`, `latitude`, `longitude`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count` y `availability_365`.
* **Target:** `log1p(price)`, para estabilizar la varianza.

---

### Eje 1: Dialéctica de Generalización (Bias, Varianza y Complejidad)

**Objetivo pedagógico:** Mostrar cómo la capacidad/complejidad del modelo altera el balance entre Underfitting y Overfitting.

**Actividad:** se divide una muestra de control y se evalúa un Random Forest con distintas profundidades máximas (1, 3, 5, 10, 20 y sin límite), comparando el RMSE de entrenamiento contra el de validación en cada caso.

#### Preguntas:

* **`max_depth = 1` (Baja complejidad / Underfitting):** Alto error tanto en Train como en Test ($Bias^2$ dominante). El modelo simplifica en exceso la geografía y dinámica de precios de Buenos Aires (ej. trata igual a Palermo que a Lugano).
* **`max_depth = 20` o `None` (Alta complejidad / Overfitting):** Error en Train cercano a cero, pero el error de Validación se dispara o estanca ($Varianza$ dominante). El árbol memorizó las excentricidades de anuncios específicos.
* **Zona Óptima:** `max_depth` entre $8$ y $12$ logra el balance ideal disminuyendo el Error Total.

---

### Eje 2: Estrategias y Esquemas de Validación

**Objetivo pedagógico:** Demostrar por qué un Hold-out simple puede ser engañoso y cómo $K$-Fold Cross-Validation refleja el rendimiento real.

**Actividad:** se define un pipeline estándar (imputación por mediana y escalado para las variables numéricas, One-Hot para las categóricas, y un modelo Ridge como línea base) y se evalúa con K-Fold Cross-Validation (K=5, con shuffle) sobre el conjunto de desarrollo, reportando la media y la desviación estándar del RMSE.

#### Discusión teórica para el aula:

* **Muestreo e Incertidumbre:** Reportar la media ($\mu$) junto con la desviación estándar ($\sigma$) del error en los 5 *folds* le permite al estudiante medir qué tan sensible es el modelo a cambios en los datos de entrada.
* **Elección del Esquema:**
  * En predicción de precios (regresión), **K-Fold con Shuffle** es la norma.
  * Se aclara que si el problema fuera predecir si un departamento es "De Lujo" (Clasificación desbalanceada, $<5\%$ del dataset), usaríamos **Stratified K-Fold** para evitar *folds* sin la clase minoritaria.

---

### Eje 3: Optimización Controlada de Hiperparámetros

**Objetivo pedagógico:** Aplicar la búsqueda de hiperparámetros dentro del bucle de validación sin violar la integridad del Test Set.

**Actividad:**

1. **Separación estricta desde el inicio:** se reserva el 20% de los datos como Test Set final.
2. **Espacio de búsqueda para Random Forest:** `n_estimators` (50, 100, 200), `max_depth` (6, 10, 15, 20, None), `min_samples_split` (2, 5, 10) y `min_samples_leaf` (1, 2, 4).
3. **Búsqueda aleatoria dentro de Cross-Validation:** 10 combinaciones evaluadas con 5 folds, usando RMSE como métrica.
4. **Evaluación única y final sobre el Test Set intacto:** se reportan el RMSE en escala log y el MAE en pesos argentinos reales.

---

### Eje 4: Anti-Patrones Críticos en la Práctica Profesional

**Objetivo pedagógico:** Identificar explícitamente cómo se comete *Data Leakage* y cómo prevenirlo estructuralmente utilizando `Pipeline`.

#### 1. Fuga por Preprocesamiento (Data Leakage)

* **El Anti-Patrón:** Imputar valores faltantes en `reviews_per_month` o escalar la variable `minimum_nights` aplicando `.fit_transform()` a todo el conjunto de datos **antes** de hacer la división de train/test o Cross-Validation.
* **Consecuencia:** La media y la desviación estándar del conjunto de prueba filtran información hacia el modelo de entrenamiento, inflando las métricas de rendimiento con un optimismo falso.
* **Solución Operativa:** Encapsular transformaciones dentro de `Pipeline` y `ColumnTransformer`, asegurando que `.fit()` se ejecute **únicamente** sobre los *train folds* de cada iteración.

#### 2. Decisiones de diseño basadas en el Test Set

* **El Anti-Patrón:** Probar variantes de variables o ajustar hiperparámetros mirando directamente el resultado en el `X_test_final`.
* **Consecuencia:** Se sobreajusta el proceso de selección al *Test Set*, convirtiéndolo en un *Validation Set* sobreutilizado y destruyendo su valor como auditoría imparcial.

#### 3. Temporalidad de Datos (Mención contextual)

* **El Anti-Patrón:** Aplicar `KFold` aleatorio en datos que contienen la variable fecha (`last_review`).
* **Corrección:** Si quisiéramos predecir demanda o precios futuros a partir de datos históricos, debemos sustituir `KFold` por `TimeSeriesSplit` o ventanas móviles (*Rolling Windows*) para respetar estrictamente la línea temporal.

---

### Resumen para Entregar a los Alumnos

| Componente | Elección en este Dataset | Justificación Técnica |
| --- | --- | --- |
| **Target ($y$)** | `log1p(price)` | Corrige el sesgo positivo (*right-skew*) extremo en precios en ARS. |
| **Validation Strategy** | 5-Fold Cross-Validation | Cuantifica la estabilidad sin desperdiciar datos de desarrollo. |
| **Preprocesamiento** | `Pipeline` + `ColumnTransformer` | Garantiza aislamiento total entre folds y previene *Data Leakage*. |
| **Optimización** | `RandomizedSearchCV` | Explora combinaciones complejas eficientemente bajo un presupuesto fijo. |
| **Métrica Final** | MAE en ARS reales | Métrica interpretable para negocio (desviación promedio en Pesos Argentinos). |
