# Análisis exploratorio de datos en Python

**Introducción**: "¿Qué es el Análisis Exploratorio de Datos? El Análisis Exploratorio de Datos (EDA) consiste en comprender los conjuntos de datos resumiendo sus características principales, a menudo trazándolas visualmente.".



* "1. Importación de las bibliotecas requeridas para el EDA".
* "2. Carga de los datos en el data frame.".
* "3. Comprobación de los tipos de datos".
* "4. Eliminación de columnas irrelevantes".
* "5. Renombrar las columnas".
* "6. Eliminación de las filas duplicadas".
* "7. Eliminación de los valores nulos o faltantes.".
* "8. Detección de valores atípicos (Outliers)".
* "9. Graficar diferentes características entre sí (dispersión) y contra la frecuencia (histograma)".





### Análisis de los puntos principales del EDA

El proceso de Análisis Exploratorio de Datos en este documento se centra fuertemente en la limpieza y adecuación del conjunto de datos:

* **Simplificación de características**: Se descartaron columnas que no aportaban valor analítico para este caso específico, tales como el tamaño del vehículo, el estilo o el tipo de combustible. Además, se renombraron columnas extensas (ej. "Engine HP" a "HP") para facilitar la lectura del código.


* **Deduplicación**: Se identificaron y eliminaron 989 filas de datos duplicados que podrían sesgar cualquier modelo futuro.


* **Manejo de valores nulos**: Se encontraron valores faltantes principalmente en las columnas de caballos de fuerza (HP) y cilindros. Dado que la cantidad de nulos era pequeña en proporción al total (alrededor de 100 nulos frente a más de 10,000 registros), se optó por eliminar estas filas directamente.


* **Control de Valores Atípicos (Outliers)**: Utilizando el método de la puntuación del Rango Intercuartílico (IQR), se identificaron y eliminaron puntos de datos anormalmente altos o bajos. Esta técnica filtró aproximadamente 1,600 filas adicionales del dataset.



### Inferencias a partir de los gráficos

Los gráficos generados en el cuaderno permiten extraer relaciones clave entre las variables:

* **Gráficos de Caja (Boxplots)**: Se utilizaron para visualizar los atributos de Precio, HP y Cilindros antes de aplicar el filtro IQR. Estos gráficos demostraron visualmente que existían vehículos con precios y potencias extremadamente por encima del promedio, representados por puntos fuera de los "bigotes" de las cajas.


* **Mapa de Calor (Heatmap)**: Este gráfico expone la matriz de correlación.


* Se infiere que el "Precio" tiene una fuerte correlación positiva con los caballos de fuerza "HP" (0.739) y una correlación menor con los "Cilindros" (0.354).


* Existe una correlación fuertemente negativa entre las variables de rendimiento de combustible ("MPG-H" y "MPG-C") frente al tamaño del motor ("Cilindros" en -0.70 y "HP" en -0.44).




* **Gráfico de Dispersión (Scatterplot)**: Al graficar HP contra Precio, los puntos dispersos reafirman visualmente la fuerte correlación positiva del mapa de calor, indicando una clara tendencia ascendente donde más HP resulta en un mayor precio de mercado.



### 4. Conclusiones y Patrones de los datos

A partir del análisis numérico y visual, se pueden establecer los siguientes patrones y conclusiones sobre el mercado automotriz en este dataset:

* **Potencia es sinónimo de Valor**: El factor principal que impulsa el precio de un automóvil es su potencia, específicamente los caballos de fuerza (HP). A medida que aumenta el caballaje, también lo hace el costo sugerido por el fabricante (MSRP).


* **El coste de la potencia es la eficiencia**: Existe un patrón inverso entre la potencia del vehículo y su economía de combustible. Los vehículos con motores más grandes (más cilindros) y mayor HP consumen significativamente más combustible, reduciendo sus métricas de MPG (millas por galón) tanto en ciudad como en carretera.


* **Calidad de Datos del Mundo Real**: El dataset original ilustra cómo los datos en bruto requieren un preprocesamiento intensivo. La reducción de las filas originales (11,914) a las filas finales después de limpieza y detección de outliers (9,191) resalta que casi el 23% de los datos originales eran duplicados, estaban incompletos o representaban anomalías extremas.