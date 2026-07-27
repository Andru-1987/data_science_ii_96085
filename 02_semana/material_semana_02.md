# Introducción a la Adquisición de Datos

## Contenido Teórico

### Introducción a la Adquisición de Datos

La adquisición de datos es el proceso de recolectar, filtrar y almacenar datos de diversas fuentes para que puedan ser utilizados en análisis, modelado y toma de decisiones. En ciencia de datos, este paso es crucial, ya que garantiza que el conjunto de datos sea adecuado y de calidad para obtener resultados significativos y confiables.

Se realizan procesos y técnicas mediante los cuales se recogen y organizan datos de diversas fuentes, como bases de datos, sitios web, APIs, sensores, redes sociales entre otras, para su posterior análisis.

### Objetivos de la Adquisición de Datos

- **Obtener datos relevantes**: Asegurarse de que los datos recolectados son apropiados para el problema que se quiere resolver.
- **Garantizar la calidad de los datos**: Evitar datos incompletos, erróneos o corruptos que puedan afectar los resultados del análisis.
- **Organizar los datos eficientemente**: Preparar los datos para que puedan ser utilizados de manera eficiente en el análisis.

### Propósitos de la Adquisición de Datos

- **Toma de decisiones informada**: Los datos permiten tomar decisiones más precisas y basadas en hechos en lugar de suposiciones.
- **Automatización y predicciones**: Modelos de machine learning y algoritmos utilizan datos para predecir tendencias y comportamientos futuros.
- **Optimización de recursos**: A través del análisis de datos, se pueden identificar ineficiencias y optimizar procesos en una organización.

### Tipos de Datos

Dependiendo de la estructura de los datos, se pueden clasificar en tres categorías principales:

**Datos estructurados**:
- Son datos organizados en tablas o bases de datos, donde cada valor tiene un formato predefinido.
- Ejemplos: hojas de cálculo, bases de datos SQL, información de ventas en una tienda (ID de producto, cantidad, precio, fecha).

**Datos no estructurados**:
- Son datos sin una estructura predefinida, como imágenes, videos, textos largos o correos electrónicos. Su análisis suele ser más complejo y requiere herramientas especializadas.
- Ejemplos: comentarios en redes sociales, correos electrónicos, fotos.

**Datos semi-estructurados**:
- Están entre los datos estructurados y no estructurados, ya que tienen algo de estructura pero no tan rigurosa como las bases de datos relacionales.
- Ejemplos: datos de una API en formato JSON, documentos XML.

### Fuentes de Datos Comunes

**Bases de datos relacionales (SQL)**:
- MySQL, PostgreSQL, SQLite
- Acceso mediante consultas en lenguaje SQL
- Ejemplo: base de datos de una empresa que almacena información de sus clientes y ventas.

**APIs (RESTful APIs, JSON)**:
- Interfaces que permiten que diferentes aplicaciones intercambien datos entre sí
- Suelen devolver los datos en formato JSON
- Ejemplo: API de clima que devuelve información meteorológica en tiempo real.

**Web Scraping**:
- Técnica para extraer datos directamente de páginas web cuando no están disponibles a través de una API o base de datos
- Implica automatizar la descarga de contenido web y transformarlo en un formato utilizable
- Ejemplo: extraer precios de productos de una tienda en línea para realizar un análisis de la competencia.

**Datos abiertos y datasets públicos**:
- Datos publicados por organizaciones gubernamentales, académicas y empresas
- Plataformas: Kaggle, sitios web de datos gubernamentales
- Ejemplo: datos de censos poblacionales, estadísticas económicas o datasets académicos.

**Dispositivos IoT, sensores y redes sociales**:
- Dispositivos y sensores generan datos continuamente para análisis en tiempo real
- Redes sociales como Twitter o Instagram son fuentes ricas de datos no estructurados
- Ejemplo: datos de tráfico capturados por sensores en una ciudad o publicaciones de usuarios en Twitter sobre un tema específico.

### Códigos de Estado HTTP Comunes

- **200**: todo salió bien y se ha devuelto el resultado (si lo hay).
- **301**: el servidor lo está redirigiendo a un punto final diferente.
- **400**: el servidor cree que hiciste una mala solicitud.
- **403**: el recurso al que intentas acceder está prohibido.
- **404**: el recurso al que intentaste acceder no se encontró en el servidor.
- **503**: el servidor no está listo para manejar la solicitud.

### Autenticación en APIs

Algunas APIs requieren autenticación para acceder a sus datos. Esto se hace mediante:
- **Claves API (API keys)**
- **OAuth**: protocolo para autenticación más complejo pero más seguro.

### Web Scraping

El web scraping es la técnica de extraer información de sitios web de manera automatizada. Es importante respetar los términos de uso de los sitios web y no sobrecargar los servidores con demasiadas solicitudes.

**Herramientas comunes**: BeautifulSoup, Scrapy, Selenium.

### Formato Pickle

El formato **Pickle** en Python se utiliza para la serialización y deserialización de objetos, permitiendo guardar y cargar estructuras de datos complejas, como DataFrames, listas y diccionarios, de manera eficiente.

**Ventajas**:
- Almacena modelos entrenados para evitar volver a entrenarlos desde cero
- Ahorra tiempo y recursos

**Alternativas**: Joblib, HDF5.
