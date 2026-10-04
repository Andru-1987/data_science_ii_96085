# Reduccion y Agrupacion 

#### Premisa central
La reducción de dimensionalidad y la agrupación de datos (*clustering*) son técnicas fundamentales del aprendizaje no supervisado enfocadas en simplificar conjuntos de datos complejos sin perder información esencial. Su propósito es mitigar la multicolinealidad, eliminar el ruido, optimizar el uso de recursos computacionales y descubrir patrones o segmentos latentes para la toma de decisiones estratégicas.

---

### Top 10 ideas clave

1. **Reducción de dimensionalidad como simplificación estructural**: Transforma datos desde espacios de alta dimensión hacia subespacios de menor dimensión reteniendo las características más significativas.
2. **Prevención del sobreajuste (*overfitting*)**: Al eliminar atributos irrelevantes o ruidosos, reduce la complejidad del modelo y mejora su capacidad de generalización ante datos no vistos.
3. **Diferencia entre Selección y Extracción de características**: La selección elige un subconjunto de las variables originales más importantes, mientras que la extracción genera variables sintéticas totalmente nuevas combinando las existentes.
4. **Fundamento del Análisis de Componentes Principales (PCA)**: Transforma variables originalmente correlacionadas en un conjunto menor de componentes ortogonales (no correlacionados) que maximizan la varianza retenida.
5. **Estandarización como requisito crítico**: Tanto PCA como K-means requieren escalar previamente las variables; de lo contrario, las características con magnitudes numéricas más grandes dominarán de forma artificial los cálculos de varianza y distancia.
6. **Agrupación no supervisada mediante K-means**: Divide un conjunto de datos no etiquetados en *K* clústeres bien definidos, asignando cada observación al centroide más cercano mediante la distancia euclidiana.
7. **Proceso iterativo de convergencia**: K-means selecciona centroides iniciales, asigna los datos al centroide más próximo, recalcula el centro promedio de cada grupo y repite este ciclo hasta que las posiciones de los centroides se estabilizan.
8. **Determinación del parámetro *K***: Para definir la cantidad óptima de grupos se utiliza el "método del codo" (*Elbow Method*), el cual analiza la variación total dentro de los clústeres.
9. **Eliminación de multicolinealidad y redundancia**: Al transformar datos correlacionados en componentes principales independientes, PCA elimina la redundancia e incrementa la estabilidad de los modelos predictivos.
10. **Técnicas avanzadas basadas en densidad**: Además de K-means, existen algoritmos como DBSCAN e HDBSCAN que identifican clústeres de formas complejas separados por zonas de menor densidad, tolerando ruido y *outliers*.

---

### Desglose por bloques

* **Bloque 1: Introducción a la Reducción de Dimensionalidad (Sección 12.1)**: Fundamenta la necesidad de simplificar grandes volúmenes de datos para disminuir los tiempos de entrenamiento, reducir el consumo de recursos computacionales, eliminar el ruido y evitar el sobreajuste. Introduce las dos grandes estrategias: Selección y Extracción de características.
* **Bloque 2: Análisis de Componentes Principales - PCA (Sección 12.2)**: Detalla el flujo matemático e iterativo del PCA: estandarización de datos, cálculo de la matriz de covarianza, obtención de autovalores y autovectores, y proyección en el nuevo subespacio. Destaca su valor para la compresión de información y la preparación de datos (*feature engineering*).
* **Bloque 3: Algoritmo K-means y Agrupamiento (Sección 12.3)**: Aborda la lógica del *clustering* no supervisado, describiendo paso a paso el algoritmo K-means, la inicialización de centroides, la reasignación por distancia euclidiana y el criterio de parada por convergencia.
* **Bloque 4: Aplicaciones Prácticas e Integración (Sección 12.4)**: Resume los casos de uso reales de estas técnicas en la industria, abarcando la segmentación de clientes, el análisis de imágenes, la clasificación de documentos, el monitoreo de redes de sensores y la detección de fraudes.

---

### Conclusiones prácticas

* **Optimización de pipelines de Machine Learning**: Incorporar PCA como paso de preprocesamiento reduce drásticamente el costo computacional y acorta los tiempos de entrenamiento de los modelos predictivos.
* **Segmentación de clientes y personalización comercial**: La implementación de K-means permite agrupar usuarios con comportamientos de compra o perfiles similares, optimizando la efectividad de las campañas de marketing y la retención.
* **Detección de anomalías y prevención de fraudes**: El *clustering* ayuda a identificar transacciones o patrones anómalos que no pertenecen a ningún grupo claramente definido, sirviendo como motor para la gestión de riesgos financieros y de ciberseguridad.
* **Comunicación visual de datos multidimensionales**: Proyectar atributos masivos en 2 o 3 componentes principales facilita crear visualizaciones claras e interpretables para presentar hallazgos complejos ante audiencias directivas. Reduccion y Agrupacion

---

## Ajustes conceptuales necesarios

Antes de la versión final, conviene matizar estas ideas:

| Tema | Ajuste recomendado |
|---|---|
| PCA y multicolinealidad | PCA transforma las variables en componentes ortogonales. Esto elimina la correlación lineal entre los componentes, pero no equivale a “resolver” siempre un problema predictivo: se gana estabilidad y compresión, aunque puede disminuir la interpretabilidad de las variables originales. |
| PCA y estandarización | PCA centra los datos, pero no los escala automáticamente por variable. Si las variables están en unidades o escalas diferentes —por ejemplo, ingresos, edad y cantidad de compras— debe usarse `StandardScaler` antes de PCA.  [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) |
| K-means y “clústeres bien definidos” | K-means funciona mejor cuando los grupos son aproximadamente compactos, de tamaño/densidad comparable y con geometría cercana a esférica en el espacio de variables escaladas. No es apropiado para todo tipo de estructura. |
| Método del codo | El codo es una heurística visual basada en la inercia; no determina por sí solo el mejor \(K\). Debe complementarse con métricas como *silhouette score*, estabilidad entre corridas e interpretación de negocio. La inercia es la suma de cuadrados intra-clúster que K-means busca minimizar.  [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) |
| PCA como “eliminación de ruido” | PCA puede ayudar a reducir ruido si se descartan componentes de baja varianza, pero no garantiza que toda componente de baja varianza sea irrelevante. Una señal crítica para negocio puede explicar poca varianza global. |
| Outliers | K-means es sensible a valores atípicos porque los centroides se calculan mediante promedios. Si hay fraude, errores de medición o colas extremas, conviene evaluar `RobustScaler`, tratamiento de outliers, DBSCAN/HDBSCAN o modelos específicos de anomalías. |
| HDBSCAN | Es correcto incluirlo como técnica avanzada. A diferencia de DBSCAN, HDBSCAN evalúa múltiples escalas de densidad y busca agrupaciones estables, por lo que suele ser más flexible cuando existen grupos con densidades distintas.  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) |

***

# Reducción de Dimensionalidad y Agrupación de Datos

## Premisa central

La **reducción de dimensionalidad** y la **agrupación de datos** (*clustering*) son técnicas fundamentales del aprendizaje no supervisado. Ambas permiten comprender conjuntos de datos complejos sin depender de una variable objetivo previamente etiquetada.

La reducción de dimensionalidad disminuye la cantidad de variables utilizadas para representar los datos, procurando retener la información estructural más relevante. El *clustering*, en cambio, identifica grupos naturales de observaciones similares entre sí y diferentes de las restantes.

Estas técnicas se usan para:

- Simplificar datasets con muchas variables.
- Reducir redundancia y correlaciones entre atributos.
- Acelerar entrenamiento, almacenamiento y visualización.
- Mitigar parte del riesgo de sobreajuste en modelos posteriores.
- Detectar segmentos, patrones, comportamientos atípicos y estructuras latentes.
- Preparar variables para modelos supervisados, sistemas de recomendación, análisis exploratorio o detección de anomalías.

> Idea central: la reducción de dimensionalidad responde “¿con qué representación más compacta puedo describir los datos?”, mientras que el *clustering* responde “¿qué grupos naturales existen dentro de los datos?”.

***

## 1. Reducción de dimensionalidad

### 1.1. ¿Qué es?

La reducción de dimensionalidad consiste en representar un dataset con menos variables o dimensiones, manteniendo la mayor cantidad posible de información útil.

Si un dataset posee \(p\) variables originales:

\[
X = [x_1, x_2, x_3, \ldots, x_p]
\]

la reducción de dimensionalidad busca construir una representación:

\[
Z = [z_1, z_2, z_3, \ldots, z_k]
\]

donde:

\[
k \ll p
\]

Por ejemplo, un dataset de clientes puede contener 80 variables: ingresos, frecuencia de compra, antigüedad, categorías preferidas, visitas al sitio, uso de descuentos, canal de compra, ubicación y métricas de interacción. PCA podría condensar parte de esa información en 5 o 10 componentes principales.

### 1.2. Problemas que ayuda a resolver

A medida que aumenta el número de variables, aparecen desafíos conocidos como la **maldición de la dimensionalidad**:

- Las distancias entre observaciones pueden perder capacidad discriminativa.
- Aumenta el costo de almacenamiento y procesamiento.
- Se necesitan más datos para estimar patrones con confiabilidad.
- El modelo puede aprender ruido o relaciones accidentales.
- Se vuelve más difícil visualizar los datos.
- Pueden aparecer variables redundantes o altamente correlacionadas.

La reducción de dimensionalidad no elimina automáticamente todos estos problemas, pero puede disminuir su impacto al producir una representación más compacta y estructurada.

### 1.3. Selección vs. extracción de características

Existen dos estrategias principales.

| Estrategia | Qué hace | Resultado | Ejemplos |
|---|---|---|---|
| Selección de características | Conserva un subconjunto de variables originales | Las columnas siguen siendo interpretables directamente | `SelectKBest`, RFE, Lasso, importancia de variables |
| Extracción de características | Crea nuevas variables a partir de combinaciones o transformaciones de las originales | Las nuevas variables pueden perder interpretabilidad directa | PCA, ICA, SVD, autoencoders, UMAP |

#### Selección de características

La selección responde a la pregunta:

> “¿Qué variables originales son más útiles para el problema?”

Ejemplo: de 50 columnas relacionadas con actividad de clientes, se seleccionan solo:

- Frecuencia de compra.
- Gasto promedio.
- Días desde la última compra.
- Antigüedad del cliente.
- Uso de descuentos.

Su ventaja es la interpretabilidad: sigue siendo posible explicar que una variable concreta influyó en un resultado.

#### Extracción de características

La extracción responde a otra pregunta:

> “¿Puedo construir una representación más eficiente combinando las variables originales?”

Por ejemplo, PCA podría generar un componente asociado a “intensidad de consumo”, combinando gasto, frecuencia, visitas y compras recientes. Ese componente puede ser muy útil para modelar, aunque sea menos intuitivo que una columna original.

***

## 2. Análisis de Componentes Principales (PCA)

### 2.1. Definición

El **Análisis de Componentes Principales** (*Principal Component Analysis*, PCA) es una técnica de extracción de características que transforma variables posiblemente correlacionadas en un conjunto de nuevas variables llamadas **componentes principales**.

Los componentes tienen dos propiedades esenciales:

1. Son ortogonales entre sí, es decir, no presentan correlación lineal.
2. Se ordenan según la cantidad de varianza que explican.

El primer componente principal captura la mayor cantidad posible de variabilidad presente en los datos. El segundo captura la mayor cantidad de variabilidad restante, bajo la restricción de ser ortogonal al primero. El proceso continúa de forma sucesiva. [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html)

### 2.2. Intuición geométrica

Imaginemos clientes descritos por dos variables:

- Ingreso mensual.
- Gasto mensual.

Es probable que ambas estén correlacionadas: a mayor ingreso, en promedio, mayor gasto. PCA puede rotar los ejes originales para encontrar una dirección que concentre la mayor variabilidad.

En lugar de analizar dos dimensiones correlacionadas, se obtienen:

- **Componente 1:** nivel económico y capacidad de gasto general.
- **Componente 2:** desviación entre ingreso y gasto, por ejemplo clientes que gastan más o menos de lo esperable según su ingreso.

Si el primer componente explica, por ejemplo, el 92% de la varianza, quizá sea suficiente conservarlo para ciertos análisis exploratorios.

### 2.3. Flujo matemático simplificado

El flujo conceptual de PCA suele incluir los siguientes pasos:

1. Recolectar una matriz de datos \(X\) de tamaño \(n \times p\), donde \(n\) es la cantidad de observaciones y \(p\) la cantidad de variables.
2. Centrar y, generalmente, estandarizar las variables.
3. Calcular la estructura de variabilidad y covarianza entre variables.
4. Obtener los autovectores y autovalores, o aplicar una descomposición equivalente basada en SVD.
5. Ordenar las direcciones principales por varianza explicada.
6. Seleccionar los primeros \(k\) componentes.
7. Proyectar los datos originales en ese nuevo subespacio.

La transformación puede expresarse de manera simplificada como:

\[
Z = XW_k
\]

donde:

- \(X\) representa los datos centrados o estandarizados.
- \(W_k\) contiene los vectores asociados con los \(k\) componentes seleccionados.
- \(Z\) es la nueva representación reducida.

### 2.4. Varianza explicada

PCA permite medir qué proporción de la variabilidad total conserva cada componente.

Por ejemplo:

| Componente | Varianza explicada | Varianza acumulada |
|---|---:|---:|
| PC1 | 48% | 48% |
| PC2 | 22% | 70% |
| PC3 | 14% | 84% |
| PC4 | 8% | 92% |
| PC5 | 4% | 96% |

En este caso, conservar cuatro componentes mantiene aproximadamente el 92% de la varianza total.

No existe un umbral universal, pero en contextos prácticos suelen evaluarse alternativas como:

- 80% de varianza acumulada cuando se prioriza compresión.
- 90% o 95% cuando se busca conservar más información.
- Un número bajo de componentes, como 2 o 3, cuando el objetivo principal es visualización.

La decisión no debe basarse solo en el porcentaje: también debe validarse el efecto sobre el objetivo final, por ejemplo la calidad de segmentación, el rendimiento de un clasificador o la capacidad de explicar el resultado a usuarios de negocio.

### 2.5. Estandarización antes de PCA

La estandarización es uno de los pasos más importantes.

PCA utiliza varianza para decidir qué direcciones son más relevantes. Si una variable tiene valores entre 0 y 1, mientras otra varía entre 0 y 1.000.000, la segunda puede dominar el resultado solo por su escala numérica, no porque sea más informativa.

La estandarización típica utiliza:

\[
z = \frac{x - \mu}{\sigma}
\]

donde:

- \(x\) es el valor original.
- \(\mu\) es la media de la variable.
- \(\sigma\) es su desvío estándar.
- \(z\) es el valor estandarizado.

`StandardScaler` centra cada variable alrededor de cero y la escala a varianza unitaria. La documentación de scikit-learn indica que esta estandarización es un requerimiento frecuente para muchos estimadores, especialmente cuando las variables tienen escalas muy diferentes. [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html)

Es importante recordar que la implementación de PCA de scikit-learn centra los datos, pero **no escala cada variable por defecto**. Por eso, cuando los atributos usan unidades distintas, el flujo habitual es:

```text
Datos originales
    ↓
StandardScaler
    ↓
PCA
    ↓
Modelo, clustering o visualización
```



### 2.6. Ventajas y limitaciones de PCA

| Ventajas | Limitaciones |
|---|---|
| Reduce la cantidad de variables | Puede dificultar la interpretación de los componentes |
| Elimina correlación lineal entre componentes | Solo captura relaciones lineales |
| Puede acelerar modelos posteriores | La varianza alta no siempre implica relevancia para negocio |
| Facilita visualización en 2D o 3D | Es sensible a escalas y, en ciertos casos, a outliers |
| Puede reducir ruido si se descartan componentes débiles | La reducción excesiva puede eliminar información predictiva importante |

***

## 3. Agrupación de datos o Clustering

### 3.1. ¿Qué es clustering?

El *clustering* es una familia de técnicas no supervisadas cuyo objetivo es dividir observaciones no etiquetadas en grupos llamados **clústeres**.

La idea es que los elementos dentro de un mismo grupo sean relativamente similares entre sí, mientras que los elementos de grupos distintos presenten diferencias relevantes.

A diferencia de la clasificación supervisada, no existe una columna objetivo como:

```text
fraude = sí / no
cliente = premium / estándar
imagen = gato / perro
```

En *clustering*, el algoritmo intenta descubrir la estructura sin disponer de etiquetas previas.

### 3.2. Ejemplo: segmentación de clientes

Supongamos un e-commerce con variables como:

- Cantidad de compras.
- Gasto acumulado.
- Ticket promedio.
- Recencia de compra.
- Porcentaje de compras con descuento.
- Categoría favorita.
- Frecuencia de visitas al sitio.

Un algoritmo de *clustering* podría descubrir segmentos como:

| Clúster | Comportamiento posible | Acción de negocio |
|---|---|---|
| 0 | Compradores frecuentes de alto valor | Beneficios de fidelización y atención preferencial |
| 1 | Clientes nuevos con bajo historial | Campañas de onboarding y primera recompra |
| 2 | Clientes sensibles a descuentos | Promociones controladas para evitar erosión de margen |
| 3 | Clientes inactivos o en riesgo de abandono | Campañas de reactivación o retención |

El nombre del clúster no lo genera el algoritmo: surge de analizar el perfil promedio de cada grupo y traducirlo a lenguaje de negocio.

***

## 4. K-means

### 4.1. Definición

K-means es uno de los algoritmos de *clustering* más utilizados. Divide los datos en \(K\) grupos, representados por un centroide.

Un centroide es, de forma simplificada, el promedio de las observaciones asignadas a un grupo.

El algoritmo busca minimizar la **inercia**, también llamada suma de cuadrados intra-clúster:

\[
\text{Inercia} = \sum_{i=1}^{n} \min_{\mu_j \in C} \left\|x_i - \mu_j\right\|^2
\]

Esto significa que intenta reducir la distancia cuadrática entre cada punto y el centroide del clúster al que pertenece. [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)

### 4.2. Funcionamiento paso a paso

El algoritmo puede describirse en cinco etapas:

1. Se define el número de clústeres \(K\).
2. Se inicializan \(K\) centroides.
3. Cada observación se asigna al centroide más cercano.
4. Se recalculan los centroides como el promedio de las observaciones asignadas.
5. Se repiten asignación y actualización hasta que los centroides dejan de cambiar de manera relevante, o se alcanza el número máximo de iteraciones.

La inicialización importa porque K-means puede converger a soluciones locales. Por ello, las implementaciones modernas suelen usar `k-means++`, que selecciona centroides iniciales de forma más informada para mejorar la convergencia. scikit-learn utiliza una variante denominada *greedy k-means++*. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

### 4.3. Distancia euclidiana

La distancia euclidiana entre dos puntos \(a\) y \(b\) puede expresarse como:

\[
d(a,b) = \sqrt{\sum_{i=1}^{p}(a_i - b_i)^2}
\]

Como K-means utiliza distancias, las variables deben estar en escalas comparables.

Por ejemplo:

- Edad: 18 a 80.
- Ingreso anual: 20.000 a 300.000.
- Número de compras: 0 a 500.

Sin escalado, el ingreso dominaría la distancia entre clientes. Por eso, el pipeline habitual es:

```text
Limpieza de datos
    ↓
Imputación de valores faltantes
    ↓
Codificación de variables categóricas
    ↓
Escalado de variables
    ↓
PCA opcional
    ↓
K-means
    ↓
Perfilado e interpretación de clústeres
```

### 4.4. Supuestos y límites de K-means

K-means es especialmente útil cuando los grupos son:

- Compactos.
- Relativamente separados.
- Aproximadamente esféricos.
- De tamaño y densidad similares.
- Descritos principalmente por variables numéricas.

No es la mejor opción cuando:

- Los clústeres tienen formas curvas, alargadas o irregulares.
- Hay densidades muy diferentes entre grupos.
- Existen muchos outliers.
- Los datos contienen variables categóricas sin una codificación y distancia adecuadas.
- No se conoce un \(K\) razonable.
- La estructura de grupos es jerárquica.

***

## 5. Cómo elegir el número de clústeres

### 5.1. Método del codo

El **método del codo** consiste en entrenar K-means con varios valores de \(K\) y graficar la inercia.

A medida que aumenta \(K\), la inercia siempre disminuye porque existen más centroides disponibles. El punto útil suele encontrarse cuando agregar más grupos produce una mejora cada vez menor.

Ejemplo conceptual:

| K | Inercia |
|---:|---:|
| 1 | 12.500 |
| 2 | 7.900 |
| 3 | 4.800 |
| 4 | 3.100 |
| 5 | 2.750 |
| 6 | 2.560 |

Si la reducción es muy pronunciada hasta \(K = 4\) y luego se aplana, podría considerarse \(K = 4\) como candidato.

Sin embargo, no siempre existe un “codo” visual claro. Por eso, esta técnica debe complementarse con otras evidencias.

### 5.2. Silhouette score

El **silhouette score** evalúa, de forma simplificada, dos aspectos:

- Cohesión: qué tan cerca está una observación de su propio clúster.
- Separación: qué tan lejos está de otros clústeres.

Su valor se interpreta aproximadamente así:

| Valor | Interpretación |
|---:|---|
| Cercano a 1 | Clústeres bien separados y cohesionados |
| Cercano a 0 | Observaciones en zonas de frontera |
| Menor que 0 | Posible asignación incorrecta a un clúster |

No debe usarse como una verdad absoluta. Un valor algo menor puede ser aceptable si la segmentación es accionable y tiene valor para el negocio.

### 5.3. Criterios de negocio

La mejor segmentación no es necesariamente la que tiene la mejor métrica matemática.

Un conjunto de cuatro segmentos puede ser preferible a uno de nueve si:

- Los grupos pueden describirse con claridad.
- Cada segmento representa una estrategia diferenciada.
- El área de negocio puede operar campañas o decisiones para esos grupos.
- La segmentación se mantiene estable cuando se incorporan nuevos datos.

***

## 6. Clustering basado en densidad

### 6.1. DBSCAN

DBSCAN identifica clústeres como regiones densas de puntos separadas por regiones de baja densidad. A diferencia de K-means:

- No exige definir \(K\) de antemano.
- Puede detectar grupos con formas arbitrarias.
- Puede marcar observaciones como ruido.
- Es útil cuando existen clusters no esféricos.

DBSCAN forma grupos a partir de muestras centrales ubicadas en regiones de alta densidad. Es especialmente adecuado cuando los clústeres tienen densidades similares y formas complejas. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

Sus parámetros principales son:

| Parámetro | Función |
|---|---|
| `eps` | Radio máximo para considerar que dos observaciones son vecinas |
| `min_samples` | Cantidad mínima de puntos necesarios para definir una región densa |

Su principal dificultad es que la elección de `eps` puede ser sensible a la escala de datos y a la densidad existente.

### 6.2. HDBSCAN

HDBSCAN es una extensión jerárquica de DBSCAN que explora múltiples niveles de densidad en lugar de fijar un único valor de `eps`.

Esto ofrece ventajas importantes:

- Puede detectar grupos de densidad variable.
- Es más robusto frente a la selección manual de parámetros.
- Identifica clústeres basándose en su estabilidad a través de distintas escalas de densidad.
- Puede tratar observaciones ambiguas como ruido.

HDBSCAN puede entenderse como la ejecución de DBSCAN sobre distintos valores de densidad, integrando después los agrupamientos más estables. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

### 6.3. Comparación de algoritmos

| Algoritmo | Requiere definir K | Detecta formas complejas | Maneja ruido | Sensible a outliers | Mejor escenario |
|---|---:|---:|---:|---:|---|
| K-means | Sí | No, generalmente | No explícitamente | Alta | Segmentos compactos y relativamente esféricos |
| DBSCAN | No | Sí | Sí | Menor que K-means | Clústeres de densidad similar con formas irregulares |
| HDBSCAN | No | Sí | Sí | Menor que K-means | Densidades variables y necesidad de robustez |
| Clustering jerárquico | No necesariamente | Depende de la métrica y enlace | No de forma nativa | Moderada | Explorar relaciones jerárquicas entre observaciones |

***

## 7. Integración de PCA y K-means

PCA y K-means suelen utilizarse juntos, pero cumplen funciones distintas.

- **PCA** comprime y transforma el espacio de variables.
- **K-means** asigna observaciones a grupos dentro de ese espacio.

Un pipeline típico sería:

```text
Datos crudos
    ↓
Limpieza e imputación
    ↓
Codificación de variables categóricas
    ↓
Escalado con StandardScaler
    ↓
PCA para retener, por ejemplo, 90–95 % de la varianza
    ↓
Evaluación de distintos valores de K
    ↓
K-means
    ↓
Silhouette score, estabilidad y perfilado de clústeres
    ↓
Visualización con PCA en 2D
    ↓
Decisiones de negocio
```

### Precaución importante

No se debe aplicar PCA automáticamente antes de K-means en todos los casos.

Puede ser útil cuando:

- Hay muchas variables numéricas.
- Existe alta correlación entre atributos.
- Se busca reducir ruido o costo computacional.
- Se necesitan visualizaciones.
- La dimensionalidad perjudica la calidad de las distancias.

Puede no ser conveniente cuando:

- Hay pocas variables y son muy interpretables.
- El negocio necesita explicar directamente cada variable.
- PCA elimina señales pequeñas pero útiles.
- La representación reducida empeora las métricas de agrupación.

La decisión correcta debe compararse experimentalmente:

1. K-means con datos escalados.
2. PCA + K-means con distintos niveles de varianza retenida.
3. Comparación de inercia, silhouette, estabilidad e interpretabilidad.

***

## 8. Casos de uso prácticos

### Segmentación de clientes

Agrupar clientes según comportamiento, valor económico, frecuencia, canal preferido o sensibilidad a descuentos.

**Resultado esperado:** campañas personalizadas, programas de fidelización, estrategias de retención y mejor asignación de presupuesto comercial.

### Detección de anomalías

Identificar observaciones que no pertenecen claramente a ningún grupo o que aparecen como ruido.

**Ejemplos:**

- Transacciones financieras inusuales.
- Accesos anómalos en ciberseguridad.
- Fallas de sensores industriales.
- Comportamientos atípicos de usuarios.
- Errores de carga o registros duplicados.

En este contexto, DBSCAN y HDBSCAN pueden resultar más naturales que K-means porque permiten etiquetar ciertos puntos como ruido.

### Análisis de documentos y embeddings

Los documentos pueden representarse como vectores de *embeddings*. Aplicar reducción dimensional y clustering permite encontrar:

- Temas recurrentes.
- Documentos similares.
- Intenciones de usuarios.
- Preguntas frecuentes.
- Vacíos en una base de conocimiento.
- Grupos de incidentes o tickets de soporte.

Para tu contexto de sistemas RAG o asistentes empresariales, este enfoque es especialmente útil: se pueden agrupar consultas históricas o fragmentos documentales para detectar dominios de conocimiento redundantes, áreas poco cubiertas y patrones de búsqueda de los usuarios.

### Imágenes y visión computacional

Una imagen puede contener miles de variables si se representa por píxeles, embeddings o atributos extraídos por una red neuronal.

PCA puede ayudar a compactar representaciones; luego, clustering puede separar patrones visuales, detectar imágenes similares o explorar clases latentes antes de disponer de etiquetas.

### Sensores e IoT

En redes de sensores, puede haber medidas de temperatura, presión, humedad, vibración, consumo eléctrico, frecuencia de fallas y ubicación.

La reducción de dimensionalidad ayuda a resumir estados operativos; el clustering puede revelar patrones de funcionamiento normal, modos de operación y posibles anomalías.

***

## 9. Recomendaciones para implementación

Para una implementación didáctica y profesional en scikit-learn, conviene seguir estas prácticas:

1. Separar la preparación de datos del algoritmo de clustering.
2. Imputar valores faltantes antes de escalar o reducir dimensionalidad.
3. Tratar con cuidado variables categóricas: `OneHotEncoder` puede aumentar mucho la dimensionalidad.
4. Aplicar `StandardScaler` antes de PCA o K-means cuando las escalas sean distintas.
5. Usar PCA con objetivos explícitos: visualización, compresión o reducción de correlación.
6. Probar varios valores de \(K\), no solo uno elegido intuitivamente.
7. Complementar el método del codo con silhouette score, estabilidad e interpretación de negocio.
8. Usar varias semillas aleatorias o `n_init` adecuado para verificar que K-means sea estable.
9. Analizar los centroides y promedios por clúster para construir perfiles interpretables.
10. Evaluar DBSCAN o HDBSCAN si hay ruido, formas irregulares o densidades variables.
11. No interpretar un clúster como una verdad causal: representa similitud estadística, no una relación de causa y efecto.
12. Versionar el pipeline completo: preprocesamiento, escalado, PCA, modelo de clustering y reglas de perfilado.

***

## Conclusiones prácticas

La reducción de dimensionalidad y el clustering deben verse como componentes de un pipeline analítico, no como algoritmos aislados.

- **PCA** permite representar datasets complejos con menos dimensiones y componentes no correlacionados linealmente, priorizando la varianza retenida. Es útil para compresión, visualización y preparación de variables, aunque puede reducir interpretabilidad. [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html)
- **K-means** es una opción eficiente y útil para segmentación cuando se esperan grupos compactos y relativamente homogéneos. Optimiza la suma de cuadrados intra-clúster o inercia. [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)
- **DBSCAN y HDBSCAN** son alternativas más apropiadas cuando los grupos tienen formas no esféricas, existe ruido o se presentan densidades diferentes. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
- **La estandarización es crítica** cuando se aplican algoritmos basados en varianza o distancia, como PCA y K-means. [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html)
- **El valor real está en la interpretación posterior**: un clúster solo es útil si puede traducirse en una acción, hipótesis, decisión, alerta o estrategia concreta.

La secuencia más clara para enseñar este tema es:

```text
Problema de alta dimensionalidad
    ↓
Selección vs. extracción de características
    ↓
Escalado de datos
    ↓
PCA y varianza explicada
    ↓
Clustering no supervisado
    ↓
K-means, inercia y selección de K
    ↓
Silhouette score e interpretación de segmentos
    ↓
DBSCAN/HDBSCAN como alternativas basadas en densidad
    ↓
Aplicación real: clientes, fraude, documentos, sensores o embeddings
```
