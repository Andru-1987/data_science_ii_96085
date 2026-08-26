# Análisis Exploratorio y Visualización de Datos

¡Hola! Si alguna vez te has enfrentado a un nuevo conjunto de datos y no sabías por dónde empezar, estás en el lugar correcto. En el mundo del Data Science, antes de crear modelos predictivos complejos, necesitamos conocer nuestros datos a fondo. A esta fase inicial la llamamos **Análisis Exploratorio de Datos (EDA)** y es, sin duda, uno de los pasos más fascinantes y críticos de cualquier proyecto.

El objetivo principal del EDA es explorar exhaustivamente los datos para descubrir patrones ocultos, detectar rarezas (anomalías) y entender la historia que nos intentan contar. Para lograrlo, nos apoyamos en dos grandes aliados: la **estadística descriptiva** y la **visualización de datos**. 

---

## 1. Lo primero es lo primero: Población vs. Muestra

Antes de hacer cuentas, hay que entender de dónde vienen nuestros datos:

*   **La Población:** Son absolutamente todos los elementos o individuos que te interesan. Por ejemplo, si quieres saber el sabor favorito de helado de todo tu país, la población son todos sus habitantes. Entrevistarlos a todos sería casi imposible (o carísimo).
*   **La Muestra:** Como no podemos tener a la población completa, tomamos un grupo más pequeño pero representativo (por ejemplo, 5,000 personas de distintas ciudades). Analizamos esta muestra para intentar "adivinar" el comportamiento de toda la población.

---

## 2. Estadística Descriptiva: Conociendo a tus datos

Cuando recibes una tabla gigantesca llena de números, lo primero que quieres hacer es resumirla para entender qué está pasando. Aquí es donde entra la estadística descriptiva, la cual nos ayuda a responder dos grandes preguntas:

### A. ¿Dónde está el centro? (Medidas de Localización)
Estas medidas nos ayudan a encontrar el valor "típico" o representativo de nuestro conjunto:

*   **Media (Promedio):** Es el clásico de clásicos. Sumas todos los valores y los divides por la cantidad total. *Ojo:* Se deja engañar fácilmente si hay valores súper altos o súper bajos (imagina promediar los sueldos de 9 personas normales y el de Elon Musk).
*   **Mediana:** Si formas a todos tus datos en una fila, del más bajito al más alto, la mediana es el que queda exactamente en el centro. Es súper confiable porque no se deja influenciar por esos valores extremos.
*   **Moda:** ¡El rey de la popularidad! Es simplemente el valor que más se repite en tu muestra.
*   **Media Recortada:** Es un truco genial donde eliminamos un pequeño porcentaje de los valores más extremos (de ambas puntas) antes de sacar el promedio, logrando un equilibrio perfecto.

### B. ¿Qué tan diferentes son entre sí? (Medidas de Dispersión)
Saber dónde está el centro no basta; necesitamos saber si todos los datos están amontonaditos y son parecidos, o si están súper esparcidos.

*   **Rango:** Es la distancia más básica: tomas tu valor más grande y le restas el más pequeño.
*   **Varianza y Desviación Estándar:** Son las estrellas de la dispersión. Te dicen, en promedio, qué tan alejados están tus datos respecto a la media. Si la desviación estándar es chiquita, tus datos son muy uniformes; si es grande, ¡tienes muchísima variedad y dispersión!

---

## 3. Visualización de Datos: ¡Viéndole la cara a los números!

A los humanos se nos da mucho mejor ver imágenes que procesar miles de números. A menudo, la mejor forma de entender un conjunto de datos es graficarlos, ya que los gráficos revelan secretos que las tablas ocultan. En Python, contamos con librerías gráficas increíbles como **Matplotlib**, **Seaborn** y **Plotly** que actúan como nuestro puente comunicacional.

Aquí te muestro cuándo usar cada tipo de gráfico y por qué son el mejor amigo de un analista:

| Tipo de Gráfico / Técnica | ¿Para qué sirve en el día a día? | ¿Por qué es vital para Machine Learning? |
| :--- | :--- | :--- |
| **Histogramas y Gráficos de Densidad** | Para ver cómo se distribuye una sola variable (Ej. conocer las edades de tus clientes o cuánto gastan). | Te ayuda a detectar si los datos están sesgados hacia un lado. Si es así, sabrás si necesitas aplicarles alguna transformación matemática antes de modelar. |
| **Diagrama de Dispersión (Scatter Plot)** | Para ver si existe relación entre dos cosas (Ej. comparar el gasto en publicidad con las ventas). | Confirma de un vistazo si hay un patrón (línea, curva, etc.), ayudándote a elegir el algoritmo predictivo correcto. |
| **Diagramas de Caja (Boxplots)** | Para comparar distribuciones entre distintos grupos (Ej. gasto promedio según zona geográfica). Es una cajita que encierra el 50% de tus datos centrales. | Es el mejor detective de *outliers* (valores atípicos raros). Detectarlos a tiempo evita que tu modelo aprenda de datos engañosos. |
| **Mapas de Calor (Heatmaps)** | Para ver cómo se relacionan *todas* las variables numéricas entre sí al mismo tiempo mediante colores. | Evita la *multicolinealidad* (cuando dos variables dicen exactamente lo mismo). Permite limpiar y simplificar el modelo. |
| **Gráficos de Barras / Tortas** | Para analizar categorías rápidamente (Ej. ver la proporción de paquetes: "Entregados", "En ruta", "Devueltos"). | Da un diagnóstico instantáneo para saber si hay "desbalance" en las categorías, un problema crítico que hay que compensar. |

---

## 4. Estadística Inferencial: Dando el gran salto

¡Llegó la hora de la verdad! Una vez que has descrito tu **muestra** y la has visualizado, usas la magia de la **probabilidad** para sacar conclusiones sobre toda la **población** total. 

Dado que tu muestra es solo un pedacito de la realidad, siempre habrá un pequeño margen de duda (incertidumbre). La estadística inferencial te permite medir exactamente esa incertidumbre para saber qué tanta confianza puedes tener al extrapolar tus descubrimientos o al validar hipótesis de tu negocio.



Basado en el libro "Statistic and Maths for Data Science", el conocimiento necesario para entender la **estadística inferencial** se construye sobre varios conceptos fundamentales que permiten ir más allá de la simple descripción de los datos.

Bases necesarias para comprenderla:

### 1. El Propósito de la Inferencia Estadística

* A diferencia de la recolección de datos y la estadística descriptiva, la estadística inferencial utiliza métodos diseñados para contribuir a realizar juicios científicos frente a la incertidumbre y a la variación.


* Su objetivo principal es utilizar técnicas que permitan obtener conclusiones (o inferencias) sobre un sistema científico en su totalidad, permitiendo ir más allá de sólo reportar los datos obtenidos.


* El profesional se enfoca en sacar conclusiones sobre una **población** (el conjunto total de observaciones o individuos de interés) a partir del análisis de una **muestra** (un subconjunto de esa población).



### 2. El Papel Crucial de la Probabilidad

* La disciplina de la probabilidad brinda la transición entre la estadística descriptiva y los métodos inferenciales.


* Los elementos de probabilidad nos permiten cuantificar la fortaleza o "confianza" en nuestras conclusiones, midiendo la incertidumbre inherente al trabajar con muestras en lugar de poblaciones completas.


* En la inferencia estadística, la muestra obtenida permite sacar conclusiones acerca de la población gracias a que esta rama utiliza ampliamente los elementos de probabilidad (un proceso de razonamiento inductivo).



### 3. Distribuciones Muestrales

* Al tomar una muestra y calcular un estadístico (como la media o el promedio de esa muestra), ese número es en realidad una variable aleatoria que dependerá de la muestra específica elegida.


* La distribución de probabilidad de un estadístico se denomina **distribución muestral**.


* Comprender las distribuciones muestrales es una de las ideas fundamentales más importantes, ya que nos permite saber cómo se comportarían los datos si tomáramos muchas muestras diferentes de la misma población, sirviendo como base teórica para el resto de la inferencia.



### 4. Las Dos Áreas Principales de la Inferencia Estadística

La inferencia estadística se divide fundamentalmente en dos grandes áreas: **estimación** y **pruebas de hipótesis**.

**A. Estimación**

* **Estimación puntual:** Es el método donde se proporciona un solo número (o punto) extraído de un conjunto de datos experimentales para estimar el parámetro real de la población.


* **Estimación por intervalo (Intervalos de Confianza):** Como un estimador puntual rara vez es exactamente igual al parámetro real, es preferible calcular un intervalo dentro del cual esperaríamos encontrar el verdadero valor del parámetro. Un intervalo más amplio nos da más confianza de que el parámetro está contenido en él (por ejemplo, un 95% o 99% de confianza), mientras que la longitud del intervalo indica la precisión de la estimación.



**B. Pruebas de Hipótesis**

* A menudo, el problema no es estimar un valor, sino tomar una decisión basada en los datos acerca de alguna conjetura del sistema.


* Una **hipótesis estadística** es una aseveración o conjetura respecto a una o más poblaciones.


* En este proceso se establece una **hipótesis nula** (que a menudo representa el "status quo") y una **hipótesis alternativa** (que suele ser la nueva conjetura a probar).


* Se utiliza un "estadístico de prueba" derivado de la muestra para determinar si hay evidencia suficiente para rechazar la hipótesis nula. Para esto, se emplea frecuentemente un **valor-P** o un nivel de significancia, que indica qué tan probable era obtener esos datos si la hipótesis nula fuera cierta.



### 5. Enfoque Clásico vs. Bayesiano

* **Método Clásico (Frecuentista):** Las inferencias se basan estrictamente en la información que se obtiene de una muestra aleatoria seleccionada de la población, asumiendo que los parámetros poblacionales son cantidades fijas pero desconocidas.


* **Método Bayesiano:** Utiliza un enfoque o "perspectiva condicional" donde los parámetros de la población no solo se manejan como desconocidos, sino como variables aleatorias. Este método permite utilizar el conocimiento subjetivo o "creencia previa" que el experimentador tiene sobre el parámetro, integrándolo con la información que proporcionan los datos reales de la muestra.


---



#### ¿Qué es el análisis exploratorio de datos (EDA)?

* El análisis exploratorio de datos (EDA) es utilizado por los científicos de datos para analizar e investigar conjuntos de datos y resumir sus características principales. Principalmente se realiza utilizando métodos de visualización de datos.
* El EDA facilita a los científicos de datos descubrir patrones, detectar anomalías, probar una hipótesis o comprobar suposiciones.
* El EDA se utiliza principalmente para proporcionar una mejor comprensión de las variables del conjunto de datos y sus relaciones.
* El EDA también puede ayudar a determinar si las técnicas estadísticas que estás considerando son apropiadas para el análisis de datos.
* Desarrolladas por el matemático estadounidense John Tukey en la década de 1970, las técnicas de EDA siguen siendo un método ampliamente utilizado en el proceso de exploración de datos en la actualidad.

#### ¿Por qué es tan importante el EDA en la ciencia de datos?

* El propósito principal del EDA es ayudarte a observar los datos antes de hacer cualquier suposición. Además de comprender mejor los patrones en los datos o detectar eventos inusuales, también te ayuda a encontrar relaciones interesantes entre las variables.
* Los científicos de datos pueden utilizar el análisis exploratorio para garantizar que los resultados que producen sean válidos y relevantes para los objetivos comerciales deseados.
* El EDA también ayuda a las partes interesadas al verificar que están haciendo las preguntas correctas.
* El EDA puede ayudar a responder preguntas sobre desviaciones estándar, variables categóricas e intervalos de confianza.
* Una vez que se completa el análisis exploratorio y se determinan las predicciones, sus características se pueden utilizar para análisis de datos más complejos o para el modelado, incluido el aprendizaje automático.

#### Herramientas de análisis exploratorio de datos

Las funciones estadísticas específicas y las técnicas que puedes realizar con las herramientas de análisis exploratorio de datos incluyen:

* Técnicas de agrupamiento (clustering) y reducción de dimensionalidad que ayudan a crear representaciones gráficas de datos de alta dimensión que contienen muchas variables.
* Visualización univariada para cada campo en el conjunto de datos sin procesar, con estadísticas resumidas.
* Visualizaciones bivariadas y estadísticas resumidas que te permiten evaluar la relación entre cada variable en el conjunto de datos y la variable objetivo que estás observando.
* Visualizaciones multivariadas para mapear y comprender las interacciones entre los diferentes campos en los datos.
* El agrupamiento K-means es un método de agrupamiento dentro del aprendizaje no supervisado, donde los puntos de datos se asignan a K grupos (número de clústeres) de acuerdo con su distancia al centro de gravedad de cada grupo. Los puntos de datos más cercanos a un centroide dado se agrupan bajo la misma categoría. El agrupamiento K-means se utiliza ampliamente en la segmentación de mercados, el reconocimiento de patrones y la compresión de imágenes.
* Los modelos predictivos, como la regresión lineal, utilizan la estadística y los datos para predecir resultados.

#### Tipos de análisis exploratorio de datos

Existen cuatro tipos de análisis exploratorio de datos:

1. **Univariado no gráfico:** Es la forma más simple de análisis de datos y los datos analizados constan de una sola variable. Como es una sola variable, no trata con causas o relaciones. El propósito principal del análisis univariado es describir los datos y encontrar patrones que existen en ellos.
2. **Bivariado gráfico:** Los métodos no gráficos no proporcionan una imagen completa de los datos. Por lo tanto, se requieren métodos gráficos. Los tipos de gráficos bivariados comunes incluyen:
* Diagramas de tallo y hojas que muestran todos los valores de los datos y la forma de la distribución.
* Histogramas, que son un gráfico de barras donde cada barra representa la frecuencia (número) o proporción (número/número total) para un rango de valores.
* Diagramas de caja (boxplots) que muestran gráficamente un resumen de cinco cifras: el mínimo, el primer cuartil, la mediana, el tercer cuartil y los valores máximos.


3. **Multivariado no gráfico:** Los datos multivariados surgen cuando hay más de una variable. Las técnicas de EDA multivariadas no gráficas muestran la relación entre dos o más variables de los datos, a menudo a través de tabulaciones cruzadas o estadísticas.
4. **Multivariado gráfico:** Los datos multivariados utilizan gráficos para mostrar relaciones entre dos o más conjuntos de datos. El gráfico más utilizado es un diagrama de barras agrupado, donde cada barra dentro de un grupo representa un nivel de cada categoría y los niveles de la otra variable.

Otros tipos generales de gráficos multivariados son:

* **Diagrama de dispersión (Scatterplot):** Se utiliza para trazar puntos de datos en los ejes horizontal y vertical para mostrar cuánto se ve afectada una variable por otra.
* **Gráfico multivariado:** Es una representación gráfica de las relaciones entre factores y una respuesta.
* **Gráfico de burbujas:** Una visualización de datos que muestra múltiples círculos (burbujas) en un dibujo bidimensional.
* **Mapa de calor (Heatmap):** Es una representación gráfica de datos donde los valores están representados por colores.




### Conclusión del Viaje
Dominar el Análisis Exploratorio de Datos garantiza que tus corazonadas y las hipótesis del negocio se validen correctamente con evidencia estadística antes de gastar recursos en algoritmos algorítmicos. Las visualizaciones actúan como el traductor definitivo: convierten volúmenes masivos e intimidantes de datos en descubrimientos claros, precisos y atractivos para cualquier persona en tu equipo.