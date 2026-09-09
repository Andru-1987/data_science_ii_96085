# Análisis Univariado y Bivariado: Guía Técnica para Data Science

## Índice de Temas

1. [8.1 Análisis Univariado: Fundamentos y Propósito](https://www.google.com/search?q=%2381-an%C3%A1lisis-univariado-fundamentos-y-prop%C3%B3sito)
2. [8.2 Análisis Bivariado: Relaciones entre Variables](https://www.google.com/search?q=%2382-an%C3%A1lisis-bivariado-relaciones-entre-variables)
3. [8.3 Profundización en Métricas Estadísticas](https://www.google.com/search?q=%2383-profundizaci%C3%B3n-en-m%C3%A9tricas-estad%C3%ADsticas)
* Desviación Estándar
* Asimetría (Skewness)
* Curtosis (Kurtosis)
* Correlaciones: Pearson, Spearman y Kendall


4. [8.4 Tips y Ejemplos Reales en Data Science](https://www.google.com/search?q=%2384-tips-y-ejemplos-reales-en-data-science)
5. [8.5 Glosario Técnico Ampliado](https://www.google.com/search?q=%2385-glosario-t%C3%A9cnico-ampliado)
6. [Referencias](https://www.google.com/search?q=%23referencias)

---

## 8.1 Análisis Univariado: Fundamentos y Propósito

El análisis univariado examina **una sola variable** de forma aislada. Su objetivo es comprender la distribución, tendencia central y dispersión de los datos. Es el primer paso obligatorio en el Exploratory Data Analysis (EDA) para detectar anomalías (outliers) y evaluar la necesidad de transformaciones antes de entrenar modelos predictivos.

* **Tendencia Central:** Media, mediana y moda. Se utilizan para encontrar el "centro de gravedad" de los datos.
* **Dispersión:** Rango, varianza, desviación estándar y rango intercuartílico (IQR). Miden qué tan alejados están los puntos de datos respecto al centro.
* **Visualización Clave:** Histogramas (distribuciones continuas), Box Plots (detección visual de outliers) y Bar Charts (frecuencias categóricas).

---

## 8.2 Análisis Bivariado: Relaciones entre Variables

Estudia **dos variables simultáneamente** para identificar dependencias empíricas, correlaciones o diferencias significativas entre grupos.

| Tipo de Variables | Técnica Principal | Gráfico Óptimo |
| --- | --- | --- |
| **Numérica vs Numérica** | Correlación (Pearson/Spearman), Regresión Lineal | Scatter plot (Diagrama de dispersión) |
| **Categórica vs Numérica** | t-test (2 grupos), ANOVA (3+ grupos) | Box plot o Violin plot agrupado |
| **Categórica vs Categórica** | Tabla de contingencia, Test Chi-cuadrado | Gráfico de barras apilado (100% stacked) |

---

## 8.3 Profundización en Métricas Estadísticas

A continuación, desglosamos las métricas fundamentales para el análisis de distribuciones y relaciones, con sus fórmulas, interpretación y aplicaciones en Python.

### Desviación Estándar (Standard Deviation)

Mide la cantidad promedio que los valores individuales difieren de la media. A diferencia de la varianza, se expresa en las mismas unidades que los datos originales.

* **Fórmula (Muestra):**

$$s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}$$


* **Interpretación (Regla Empírica):** Si los datos siguen una distribución normal, el ~68% de los datos cae dentro de $\bar{x} \pm 1s$, el ~95% dentro de $\bar{x} \pm 2s$ y el ~99.7% dentro de $\bar{x} \pm 3s$.
* **Caso de Uso Real:** En finanzas, la desviación estándar de los retornos diarios de un activo se utiliza como medida de **volatilidad** (riesgo). En control de calidad, si una máquina envasadora de 1 litro tiene $s = 5$ ml, calibrar la máquina para rechazar envases fuera de $\pm 3s$ (15 ml) garantiza un 99.7% de precisión.

### Asimetría (Skewness)

Evalúa la falta de simetría de una distribución respecto a su media. Indica hacia dónde se alarga la "cola" de los datos.

* **Fórmula (Coeficiente de Fisher-Pearson):**

$$g_1 = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{s^3}$$


* **Interpretación de Valores:**
* **$[-0.5, 0.5]$:** Distribución aproximadamente simétrica (ej. altura humana).
* **$> 1$ (Asimetría Positiva / Right-skewed):** La cola se extiende hacia la derecha. La media es mayor que la mediana.
* **$< -1$ (Asimetría Negativa / Left-skewed):** La cola se extiende hacia la izquierda. La mediana es mayor que la media.


* **Caso de Uso Real:** La distribución de salarios en una empresa o precios de propiedades casi siempre tiene un *skew* positivo extremo ($> 2.0$), porque pocas personas/propiedades tienen valores astronómicamente altos. **Impacto en ML:** Modelos como la regresión lineal asumen normalidad; un alto skewness exige aplicar transformaciones (ej. $log(x)$ o Box-Cox) antes de entrenar.

### Curtosis (Kurtosis)

Mide el "peso" de las colas y qué tan puntiaguda es la distribución en comparación con una curva normal. En Data Science, generalmente usamos el **Exceso de Curtosis**.

* **Fórmula (Exceso de Curtosis):**

$$g_2 = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{s^4} - 3$$


* **Interpretación de Valores:**
* **$= 0$ (Mesocúrtica):** Comportamiento idéntico a una distribución normal.
* **$> 0$ (Leptocúrtica):** Distribución con un pico pronunciado y **colas pesadas** (más outliers de lo esperado).
* **$< 0$ (Platicúrtica):** Distribución aplanada y **colas ligeras** (menos outliers).


* **Caso de Uso Real:** En gestión de riesgos (Data Science financiero), una curtosis alta (leptocúrtica) es una alerta roja. Significa que los eventos extremos (como caídas drásticas del mercado) ocurren con mayor frecuencia de lo que predeciría una distribución normal (el fenómeno del "Cisne Negro").

### Correlaciones: Pearson, Spearman y Kendall

Existen múltiples formas de medir la asociación entre dos variables numéricas, dependiendo de la naturaleza de los datos.

#### 1. Correlación de Pearson ($r$)

* **Qué mide:** Relación **lineal** entre dos variables continuas.
* **Cálculo:** Covarianza de X e Y, dividida por el producto de sus desviaciones estándar.
* **Valores:** $-1$ (inversa perfecta) a $1$ (directa perfecta). $0$ indica que no hay relación *lineal*.
* **Caso de Uso:** Relación entre metros cuadrados de una casa y su precio. Funciona bien solo si los datos no tienen outliers extremos.

#### 2. Correlación de Spearman ($\rho$)

* **Qué mide:** Relación **monótona** (si una sube, la otra también, sin importar si es una línea recta o una curva).
* **Cálculo:** Es el coeficiente de Pearson aplicado a los *rangos* (posiciones ordenadas) de los datos, no a sus valores reales.
* **Caso de Uso:** Si el crecimiento es exponencial (ej. años de experiencia vs. salario en un nicho altamente demandado), Pearson subestimará la relación, pero Spearman detectará un $\rho$ cercano a $1$. Es extremadamente robusto contra outliers.

#### 3. Correlación de Kendall ($\tau$)

* **Qué mide:** Concordancia entre pares de datos ordinales o continuos.
* **Cálculo:** $\tau = \frac{C - D}{C + D}$ (donde C son los pares concordantes y D los discordantes).
* **Caso de Uso:** Muy utilizado en sistemas de recomendación y motores de búsqueda para comparar listas de resultados. Si un experto rankea 10 películas y tu algoritmo rankea las mismas 10, usas Kendall Tau para medir qué tan similares son ambos rankings. Se prefiere sobre Spearman cuando el tamaño de la muestra es pequeño o hay muchos valores empatados.

---

## 8.4 Tips y Ejemplos Reales en Data Science

* **Tip 1: No confíes ciegamente en correlaciones bajas.**
Una correlación de Pearson igual a $0.05$ no significa que las variables sean independientes. Podrían tener una relación cuadrática perfecta (ej. $y = x^2$). **Regla:** Siempre grafica el *scatter plot* antes de descartar una variable.
* **Tip 2: La paradoja de Simpson.**
Una tendencia agregada puede desaparecer o invertirse al agrupar los datos por una variable categórica oculta. *Ejemplo:* Al analizar eficacia de dos medicamentos, el fármaco A parece mejor en general, pero al desglosar por severidad (Leve/Grave), el fármaco B es mejor en ambos subgrupos. Segmentar es vital.
* **Tip 3: Filtrado de Features (Feature Selection) con Bivariado.**
En la preparación para Machine Learning, si dos variables predictoras (ej. `impuestos_anuales` y `valor_propiedad`) tienen una correlación $> 0.85$ entre sí (multicolinealidad), los modelos de regresión lineal se vuelven inestables. Debes eliminar una de las dos.

---

## 8.5 Glosario Técnico Ampliado

* **ANOVA (Analysis of Variance):** Prueba estadística utilizada para determinar si existen diferencias estadísticamente significativas entre las medias de tres o más grupos independientes.
* **Colinealidad / Multicolinealidad:** Condición en la que dos o más variables predictoras en un modelo estadístico están altamente correlacionadas, lo que dificulta aislar el efecto individual de cada una.
* **Cuartiles / Percentiles:** Medidas de posición que dividen un conjunto de datos ordenado en partes iguales. El primer cuartil (Q1) es el percentil 25, la mediana (Q2) es el percentil 50.
* **EDA (Exploratory Data Analysis):** Enfoque analítico propuesto por John Tukey para resumir las características principales de los conjuntos de datos, a menudo utilizando métodos visuales, antes del modelado formal.
* **Grados de libertad (Degrees of Freedom):** Número de valores en el cálculo final de una estadística que son libres de variar. En una muestra de tamaño $n$ para calcular una varianza, es $n-1$.
* **Heterocedasticidad:** Ocurre cuando la dispersión (varianza) de una variable no es constante a lo largo de los valores de otra variable. Se visualiza en un scatter plot como una forma de "cono".
* **Hipótesis Nula ($H_0$):** La afirmación por defecto de que no existe relación, efecto o diferencia significativa entre grupos o variables.
* **Imputación:** El proceso estadístico de reemplazar datos faltantes (Missing Values) con valores sustitutos (como la mediana) para retener la mayor cantidad de datos posible.
* **IQR (Interquartile Range):** Diferencia entre el tercer y el primer cuartil ($Q3 - Q1$). Mide la dispersión del 50% central de los datos.
* **Outlier (Valor atípico):** Punto de datos que difiere significativamente de otras observaciones, ya sea por error de medición o por ser una rareza estadística genuina.
* **P-valor (p-value):** Probabilidad de obtener un resultado al menos tan extremo como el observado, asumiendo que la Hipótesis Nula es cierta. Generalmente, un p-valor $< 0.05$ se considera estadísticamente significativo.
* **R-cuadrado ($R^2$):** Coeficiente de determinación que representa la proporción de la varianza en la variable dependiente que es predecible a partir de la(s) variable(s) independiente(s).

---

## Referencias

1. **Tukey, J. W. (1977).** *Exploratory Data Analysis*. Addison-Wesley. (Referencia fundamental para Box Plots y filosofía EDA).
2. **Bruce, P., Bruce, A., & Gedeck, P. (2020).** *Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python*. O'Reilly Media.
3. **Wickham, H., & Grolemund, G. (2016).** *R for Data Science*. O'Reilly Media. (Capítulos de visualización univariada y bivariada).
4. Scipy Documentation: `scipy.stats.skew`, `scipy.stats.kurtosis`, `scipy.stats.pearsonr`. Disponibles en: docs.scipy.org
5. Pandas Documentation: Exploratory data analysis. Disponibles en: pandas.pydata.org