## Desafío: Optimización de Logística Urbana en Buenos Aires

### Consigna

Sos el analista principal de datos en una app de envíos rápidos que opera en la Ciudad Autónoma de Buenos Aires (CABA) y sus alrededores. El equipo directivo te pide un informe integral para entender por qué bajó el rendimiento en la Zona Sur, visualizar la cobertura espacial de los repartos sobre el mapa porteño y descubrir estadísticamente qué factores impactan más en las demoras y en la satisfacción de los clientes.

Tenés a disposición un dataset con 1500 registros de entregas recientes (medidas rigurosamente en **kilómetros**). Deberás aplicar técnicas de storytelling visual, análisis geoespacial y estadística para presentar tus hallazgos.

### Tareas a realizar

1. **Storytelling (Reporte Estático):** Armar un gráfico de barras narrativo que explique la caída del volumen de entregas en la Zona Sur durante el último mes debido a un corte prolongado en el Puente Pueyrredón, dejando el mensaje clave anotado directamente en el gráfico.
2. **Storytelling (Dashboard Interactivo):** Crear un gráfico de líneas interactivo con un menú desplegable (dropdown) que permita al gerente de operaciones revisar el tiempo promedio de entrega mes a mes, filtrando por zona de Buenos Aires (Norte, Sur, Caballito/Centro, etc.).
3. **Análisis Espacial (Mapa GIS):** Generar un mapa interactivo de burbujas centrado en CABA. El tamaño de la burbuja debe representar el costo del envío (ARS) y el color debe indicar el nivel de satisfacción del cliente (1 a 5).
4. **Análisis Estadístico (Univariado):** Analizar la distribución de la **distancia de los trayectos en kilómetros** calculando la media y el desvío estándar, y visualizarlo mediante un histograma con la media y mediana marcadas.
5. **Análisis Estadístico (Bivariado y Multivariado):** Evaluar la correlación entre la distancia (km), el tiempo de entrega (min), el costo (ARS) y la satisfacción del cliente usando un mapa de calor (heatmap).

