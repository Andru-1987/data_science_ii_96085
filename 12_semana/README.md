# Reduccion de Dimensionalidad y Clustering

La **reducción de dimensionalidad** y el **clustering** (agrupación) son los pilares del aprendizaje no supervisado. Su objetivo es tomar conjuntos de datos masivos y complejos, simplificarlos y revelar los patrones o grupos naturales que se esconden en ellos, sin necesidad de contar con datos previamente etiquetados.

## 1. Reducción de Dimensionalidad

A medida que sumamos variables a un dataset, nos enfrentamos a la "maldición de la dimensionalidad": los modelos se vuelven lentos, requieren más memoria y corren el riesgo de memorizar ruido (*overfitting*). Para solucionarlo, existen dos caminos:

* **Selección de características:** Elegir un subconjunto de las variables originales más importantes (ej. conservar solo "Gasto Promedio" y descartar el resto). Mantiene la interpretabilidad.
* **Extracción de características:** Crear variables sintéticas nuevas combinando las originales. Se pierde interpretabilidad directa, pero se comprime mejor la información.

### Análisis de Componentes Principales (PCA)

Es la técnica estrella de extracción. Transforma variables correlacionadas en nuevos ejes llamados **componentes principales**, los cuales son completamente independientes entre sí (ortogonales).

* **Cómo funciona:** El primer componente captura la mayor cantidad de variabilidad (información) posible; el segundo captura la variabilidad restante, y así sucesivamente.
* **Regla de oro:** Es **obligatorio** estandarizar los datos antes (por ejemplo, con `StandardScaler`). Si no lo haces, las variables con números más grandes (como "Ingreso Anual") dominarán a las más pequeñas (como "Edad") simplemente por su escala.

## 2. Agrupación de Datos (Clustering)

El clustering agrupa observaciones para que los elementos de un mismo grupo sean muy similares entre sí, y muy distintos a los de otros grupos.

### K-means: El estándar de la industria

Divide los datos en **K** grupos esféricos. Funciona calculando "centroides" (el centro de cada grupo) y asignando cada dato al centroide más cercano usando la distancia euclidiana.

* **Puntos débiles:** Tienes que decirle cuántos grupos (K) buscar. Además, es muy sensible a los valores atípicos (*outliers*) y asume que todos los grupos tienen tamaños y densidades similares.
* **¿Cómo elegir K?**
* *Método del codo:* Grafica la inercia (varianza interna de los grupos) y busca el punto donde la curva se aplana.
* *Silhouette Score:* Mide qué tan bien separado está cada grupo.
* *Criterio de negocio:* La cantidad de grupos debe tener sentido práctico y ser accionable para la empresa.



### Alternativas Avanzadas: Basadas en Densidad

Cuando los datos tienen formas raras, densidades distintas o mucho ruido, K-means se queda corto.

| Algoritmo | ¿Pide definir K? | Maneja ruido/outliers | Mejor caso de uso |
| --- | --- | --- | --- |
| **K-means** | Sí | No | Grupos redondos, bien separados y de tamaño similar. |
| **DBSCAN** | No | Sí | Grupos con formas irregulares pero con densidad constante. |
| **HDBSCAN** | No | Sí | Grupos de formas irregulares con densidades muy variables. |

## 3. El Pipeline Analítico Ideal

PCA y K-means suelen ser un gran equipo, pero deben ejecutarse en un orden lógico para evitar errores matemáticos. El flujo de trabajo profesional recomendado es:

1. **Limpieza:** Imputar valores nulos y codificar variables categóricas.
2. **Estandarización:** Aplicar `StandardScaler` para igualar el peso de las variables.
3. **Reducción (PCA):** Opcional. Reducir dimensiones reteniendo un 90-95% de la varianza total.
4. **Búsqueda de K:** Evaluar distintos números de grupos con el Método del Codo y Silhouette Score.
5. **Clustering:** Aplicar K-means (o HDBSCAN si hay mucho ruido).
6. **Perfilado:** Analizar los centroides para darle un "nombre" humano a cada clúster.

## 4. Casos de Uso Reales

Estas herramientas no viven en el vacío; son motores de decisiones estratégicas en múltiples industrias:

* **Marketing y E-commerce:** Segmentar clientes (ej. "Cazadores de ofertas" vs. "Compradores premium") para hiper-personalizar campañas.
* **Ciberseguridad y Finanzas:** Detectar fraudes identificando transacciones que el clustering etiqueta como ruido o que quedan muy lejos de los grupos normales.
* **Procesamiento de Lenguaje (NLP):** Agrupar documentos o tickets de soporte por similitud semántica para encontrar temas recurrentes o vacíos de información.
* **Internet de las Cosas (IoT):** Monitorear sensores industriales para definir perfiles de "funcionamiento normal" y disparar alertas tempranas de fallas.

El valor de estas técnicas no está en la matemática perfecta, sino en su capacidad para traducir un caos de números en segmentos accionables que un equipo de negocio pueda entender y utilizar.


## Ajustes conceptuales necesarios

| Tema | Ajuste recomendado |
|---|---|
| PCA y multicolinealidad | PCA transforma las variables en componentes ortogonales. Esto elimina la correlación lineal entre los componentes, pero no equivale a “resolver” siempre un problema predictivo: se gana estabilidad y compresión, aunque puede disminuir la interpretabilidad de las variables originales. |
| PCA y estandarización | PCA centra los datos, pero no los escala automáticamente por variable. Si las variables están en unidades o escalas diferentes —por ejemplo, ingresos, edad y cantidad de compras— debe usarse `StandardScaler` antes de PCA.  [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) |
| K-means y “clústeres bien definidos” | K-means funciona mejor cuando los grupos son aproximadamente compactos, de tamaño/densidad comparable y con geometría cercana a esférica en el espacio de variables escaladas. No es apropiado para todo tipo de estructura. |
| Método del codo | El codo es una heurística visual basada en la inercia; no determina por sí solo el mejor \(K\). Debe complementarse con métricas como *silhouette score*, estabilidad entre corridas e interpretación de negocio. La inercia es la suma de cuadrados intra-clúster que K-means busca minimizar.  [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) |
| PCA como “eliminación de ruido” | PCA puede ayudar a reducir ruido si se descartan componentes de baja varianza, pero no garantiza que toda componente de baja varianza sea irrelevante. Una señal crítica para negocio puede explicar poca varianza global. |
| Outliers | K-means es sensible a valores atípicos porque los centroides se calculan mediante promedios. Si hay fraude, errores de medición o colas extremas, conviene evaluar `RobustScaler`, tratamiento de outliers, DBSCAN/HDBSCAN o modelos específicos de anomalías. |
| HDBSCAN | Es correcto incluirlo como técnica avanzada. A diferencia de DBSCAN, HDBSCAN evalúa múltiples escalas de densidad y busca agrupaciones estables, por lo que suele ser más flexible cuando existen grupos con densidades distintas.  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) |



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


### Conclusiones prácticas

* **Optimización de pipelines de Machine Learning**: Incorporar PCA como paso de preprocesamiento reduce drásticamente el costo computacional y acorta los tiempos de entrenamiento de los modelos predictivos.
* **Segmentación de clientes y personalización comercial**: La implementación de K-means permite agrupar usuarios con comportamientos de compra o perfiles similares, optimizando la efectividad de las campañas de marketing y la retención.
* **Detección de anomalías y prevención de fraudes**: El *clustering* ayuda a identificar transacciones o patrones anómalos que no pertenecen a ningún grupo claramente definido, sirviendo como motor para la gestión de riesgos financieros y de ciberseguridad.
* **Comunicación visual de datos multidimensionales**: Proyectar atributos masivos en 2 o 3 componentes principales facilita crear visualizaciones claras e interpretables para presentar hallazgos complejos ante audiencias directivas. Reduccion y Agrupacion
