# Guía de Clase: Storytelling, GIS y Taxonomía del Análisis Estadístico

## 1. Storytelling Aplicado a Data Science

El storytelling en ciencia de datos se define como el arte de traducir análisis de datos complejos en narrativas comprensibles orientadas a influir en la toma de decisiones. Su objetivo es inyectar emociones a los descubrimientos de datos (insights) para transmitir un mensaje de forma efectiva a una audiencia específica.

Una narrativa visual exitosa requiere de una estructura clásica (inicio, nudo y desenlace) y debe responder siempre a tres preguntas esenciales: ¿Por qué?, ¿Quién? y ¿Qué? queremos transmitir.

### Tabla Matriz: Enfoques de Presentación en Storytelling

| Formato de Narrativa | Descripción | Ventajas (Pros) | Desventajas (Cons) | Ejemplo Simple |
| --- | --- | --- | --- | --- |
| **Reporte Estático** | Gráficos fijos explicados verbalmente por un presentador. | Alto control de la narrativa por parte del expositor; fácil de compartir. | Nula interactividad; no permite que el usuario explore los datos por su cuenta. | Presentación mensual en PDF sobre la caída de ventas en el sector norte. |
| **Dashboard Interactivo** | Tableros de control en tiempo real (ej. Power BI, Plotly) donde el usuario aplica filtros. | Otorga libertad y control al usuario; altamente eficiente para consultas rápidas. | Riesgo de sobrecargar al usuario con ruido visual si no se aplican principios de usabilidad. | Tablero de Recursos Humanos donde el gerente filtra la rotación de personal por departamento. |
| **Scrollytelling** | Visualización interactiva web que cambia dinámicamente conforme el usuario hace scroll. | Altamente inmersivo; excelente conexión emocional y recordabilidad. | Desarrollo técnico costoso y complejo (requiere programación web avanzada). | Artículo interactivo sobre cómo la deforestación avanza año tras año. |

---

## 2. Análisis Espacial y Mapas GIS

El Análisis Espacial consiste en el uso de herramientas específicas para examinar datos geográficos que cuentan con una ubicación explícita en la Tierra. Esto permite identificar patrones y comportamientos espaciales que en un análisis tabular tradicional (como un Excel) serían invisibles. Para posicionar los datos correctamente, se utilizan Sistemas de Referencia de Coordenadas (CRS), siendo el WGS 84 el estándar global más utilizado.

### Tabla Matriz: Modelos de Datos Geoespaciales

| Modelo Espacial | Geometría / Estructura | Ventajas (Pros) | Desventajas (Cons) | Ejemplo Simple |
| --- | --- | --- | --- | --- |
| **Vectorial (Puntos)** | Ubicaciones exactas mediante coordenadas. | Excelente precisión para representar entidades discretas. | No es apto para mostrar variables continuas en el terreno. | Coordenadas exactas de sucursales o postes de luz. |
| **Vectorial (Líneas)** | Objetos alargados de una sola dimensión. | Ideal para analizar redes, flujos y distancias logísticas. | El procesamiento puede ser pesado en redes urbanas muy densas. | Trazado de carreteras, ríos o rutas de vuelo. |
| **Vectorial (Polígonos)** | Áreas cerradas de dos dimensiones. | Permite calcular superficies y agrupar datos sociodemográficos. | Requiere un manejo topológico cuidadoso en las fronteras. | Límites territoriales de un país, una comuna o un parque. |
| **Ráster** | Rejilla de celdas de igual tamaño (píxeles) con valores continuos. | Ideal para representar gradientes ambientales y variables naturales fluidas. | Archivos computacionalmente pesados; pérdida de resolución al hacer zoom. | Imágenes satelitales que muestran variaciones de temperatura o altitud. |

---

## 3. Taxonomía del Análisis Estadístico de Datos

En ciencia de datos, el análisis se clasifica estrictamente en función del número de variables bajo estudio simultáneo. A medida que aumentamos las dimensiones, pasamos de describir una propiedad aislada a modelar estructuras complejas.

### Tabla Matriz: Análisis Univariado, Bivariado y Multivariado

| Nivel de Análisis | Definición Matemática | Ventajas (Pros) | Desventajas (Cons) | Ejemplo Simple |
| --- | --- | --- | --- | --- |
| **Univariado ($d=1$)** | Se enfoca de manera aislada en una única variable, utilizando medidas como la Media ($\mu$), Mediana y Varianza ($S^2$). | Diagnóstico rápido de la distribución intrínseca y detección de asimetrías. | No permite evaluar relaciones, causas, efectos ni dependencias con el entorno. | Analizar mediante un Histograma la distribución de edades de los clientes. |
| **Bivariado ($d=2$)** | Estudia la relación simultánea y los patrones de variación conjunta entre dos variables (ej. Covarianza $\sigma_{XY}$). | Facilita el descubrimiento de correlaciones y la prueba visual de hipótesis. | Asumir correlación como causalidad directa; ignora el impacto de terceras variables ocultas. | Usar un Diagrama de Dispersión (Scatter Plot) para ver cómo el presupuesto de marketing impacta las ventas. |
| **Multivariado ($d \ge 3$)** | Examina interacciones y modelación simultánea de tres o más variables. | Permite cuantificar interacciones complejas, descubrir estructuras latentes (PCA) y optimizar variables. | Complejidad matemática y metodológica alta; riesgo de multicolinealidad. | Usar un Mapa de Calor (Heatmap) para ver la correlación entre precio, metros cuadrados y ubicación de una vivienda. |

---

### Alertas Críticas para el Aula (Para debatir con los alumnos)

Para garantizar la correcta asimilación del contenido técnico, es fundamental repasar estas trampas comunes durante la clase:

1. **La trampa de la Covarianza y la Independencia:** Que la covarianza estadística entre dos variables dé cero no significa que sean independientes; simplemente indica que no existe una relación de tipo lineal (podría haber una relación parabólica perfecta).
2. **El peligro de las Escalas en PCA (Análisis Multivariado):** Si se aplica el Análisis de Componentes Principales sin estandarizar previamente los datos, la variable con la mayor escala absoluta (ej. peso en gramos vs estatura en metros) dominará todo el modelo algorítmico de manera artificial.
3. **El denominador de la Varianza Muestral:** Al calcular la varianza de una muestra, se divide por $n-1$ y no por $n$. Esto se hace para corregir matemáticamente el sesgo de estimación al haber consumido un grado de libertad al usar la media muestral.