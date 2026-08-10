## APIs y Data Wrangling

En Data Science, la obtención de los datos es solo el punto de partida. El verdadero desafío radica en transformar datos crudos y heterogéneos en información confiable y estructurada para su posterior análisis o uso en modelos de Machine Learning.

---

### 1. APIs: Conectando Sistemas

Una **API (Application Programming Interface)** actúa como un intermediario o puente que permite a dos sistemas comunicarse entre sí, ocultando la complejidad interna de cada uno. Para un Científico de Datos, las APIs son fuentes vitales de información externa (clima, mapas, finanzas, redes sociales).

**El ciclo de comunicación básico:**

> **Cliente (Data Scientist)** ➔ *Envía un Request (Petición)* ➔ **API** ➔ *Devuelve un Response (Respuesta)* ➔ **Cliente**

* **Métodos HTTP:** Definen la acción que queremos realizar. El más utilizado en la extracción de datos es `GET` (obtener información).
* **Códigos de Estado:** Indican el resultado de la petición (ej. `200` = Éxito, `401` = No autenticado, `404` = No encontrado, `429` = Límite de peticiones excedido).
* **Autenticación y Límites:** Existen APIs públicas (acceso libre) y privadas (requieren una *API Key*). Además, suelen tener límites de uso (*Rate Limits*) que exigen diseñar extracciones controladas.

### 2. JSON: El Lenguaje de Intercambio

Las APIs suelen devolver la información en formato **JSON** (JavaScript Object Notation). Es un formato semi-estructurado, ligero y flexible, basado en pares de `clave ➔ valor`. Su capacidad de anidar información lo hace ideal para el transporte de datos en la web, aunque requiere ser aplanado y transformado (generalmente a un formato tabular como un *DataFrame*) para su análisis.

---

### 3. Data Wrangling (o Data Munging)

Es el proceso sistemático de **limpieza, transformación y unificación** de datos complejos y desordenados para hacerlos utilizables. Los datos del mundo real suelen tener valores faltantes, duplicados, inconsistencias o formatos erróneos.

El flujo de vida de los datos sigue esta lógica simplificada:

> **Fuentes (APIs, CSV, SQL)** ➔ **Ingestión** ➔ **Data Wrangling** ➔ **Dataset Limpio** ➔ **Machine Learning / Análisis**

#### Las 6 Etapas del Data Wrangling:

1. **Descubrimiento:** Explorar y entender los datos antes de tocarlos. Conocer sus tipos, volumen y problemas evidentes.
2. **Estructuración:** Homogeneizar los formatos y tipos de datos (ej. asegurar que todas las fechas tengan el mismo formato y los precios sean numéricos).
3. **Limpieza:** Tratar valores faltantes (eliminar o imputar), remover duplicados reales, normalizar textos (mayúsculas/minúsculas) y evaluar valores atípicos (*outliers*).
4. **Enriquecimiento:** Cruzar y agregar información externa que aporte valor (ej. sumar datos demográficos a un listado de ventas).
5. **Validación:** Comprobar mediante reglas lógicas que las transformaciones aplicadas no introdujeron errores y que los datos mantienen su integridad.
6. **Publicación:** Almacenar o exportar el dataset final limpio para que sea consumido por analistas o modelos de Machine Learning.

---

### 4. Herramientas Teóricas de Transformación

Durante la preparación de los datos, es común utilizar operaciones relacionales y estructurales (generalmente a través de herramientas como Pandas):

* **Integración de datos:**
* **Merge (Unión por claves):** Similar a un `JOIN` en bases de datos. Combina distintas tablas de información buscando coincidencias a través de una columna en común (ej. ID de cliente).
* **Concat (Apilamiento):** Une tablas estructuralmente idénticas, apilándolas unas sobre otras (ej. unir los reportes de enero, febrero y marzo).


* **Reshaping (Cambio de forma):** Transformar la estructura de la tabla, pasando de un formato "ancho" (muchas columnas) a uno "largo" (más filas, menos columnas) o viceversa, para facilitar el análisis.
* **Agregación:** Agrupar datos por categorías y calcular métricas resumidas (sumas, promedios, conteos).

---

### 5. Buenas Prácticas

* **Conocer el dominio:** Las decisiones técnicas (cómo tratar un valor en cero o un extremo) deben basarse en el conocimiento del negocio.
* **Versionado:** Nunca sobrescribir los datos originales (*raw data*). Mantener un historial de los datos limpios.
* **Diccionario de Datos:** Documentar qué significa cada variable, su tipo de dato y su origen para estandarizar el conocimiento del equipo.

---

### Preguntas Clave para el Data Wrangling

Para asegurar un proceso de descubrimiento, limpieza y validación riguroso, un Data Scientist debe hacerse las siguientes preguntas frente a un nuevo conjunto de datos:

| Categoría | Preguntas que debes hacerte |
| --- | --- |
| **Origen y Contexto** | • ¿De dónde provienen estos datos (API, SQL, archivo local)?<br>

<br>• ¿Qué sistema o proceso los generó?<br>

<br>• ¿Tenemos un diccionario de datos o conocimiento del negocio para interpretarlos? |
| **Estructura y Volumen** | • ¿Cuántos registros (filas) y variables (columnas) existen?<br>

<br>• ¿Qué representa exactamente cada fila del dataset?<br>

<br>• ¿Qué tipos de datos contiene (texto, números, fechas) y coinciden con lo esperado? |
| **Calidad de los Datos** | • ¿Existen valores faltantes o nulos? Si es así, ¿por qué faltan y cómo deben tratarse?<br>

<br>• ¿Hay registros duplicados? ¿Son duplicados reales o coincidencias válidas?<br>

<br>• ¿Existen variables relevantes que estén anidadas o requieran normalización de texto? |
| **Validez y Outliers** | • ¿Existen valores extremos (*outliers*)? ¿Son errores de carga o anomalías válidas de la realidad?<br>

<br>• ¿Los rangos de valores tienen sentido lógico (ej. que una edad no sea negativa o un id sea único)? |
| **Enriquecimiento** | • ¿Este dataset responde al problema por sí solo o necesito cruzarlo con fuentes externas? |
| **Validación Final** | • Tras las transformaciones, ¿el volumen de datos y los cálculos agregados siguen siendo correctos y lógicos? |

---


## Resumen de conceptos principales

| Concepto                | ¿Qué debemos entender?                                                               |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **API**                 | Permite la comunicación entre sistemas y el acceso a datos/funcionalidades externas. |
| **HTTP Request**        | Solicitud que realizamos a una API.                                                  |
| **HTTP Response**       | Respuesta recibida desde la API.                                                     |
| **JSON**                | Formato semi-estructurado habitual para intercambiar datos.                          |
| **Requests**            | Librería Python para realizar solicitudes HTTP.                                      |
| **Pandas**              | Herramienta principal para manipular y analizar datos en Python.                     |
| **DataFrame**           | Estructura tabular utilizada para trabajar con los datos.                            |
| **Data Wrangling**      | Proceso de preparar, limpiar, transformar y unificar datos.                          |
| **Descubrimiento**      | Comprender los datos antes de modificarlos.                                          |
| **Estructuración**      | Llevar los datos a formatos y tipos coherentes.                                      |
| **Limpieza**            | Corregir errores, duplicados, faltantes e inconsistencias.                           |
| **Enriquecimiento**     | Incorporar información adicional.                                                    |
| **Validación**          | Comprobar que los datos siguen siendo correctos.                                     |
| **Publicación**         | Dejar los datos preparados para su consumo.                                          |
| **`merge()`**           | Combinar DataFrames mediante claves; similar a SQL `JOIN`.                           |
| **`concat()`**          | Apilar DataFrames.                                                                   |
| **`groupby()`**         | Agrupar datos para obtener información resumida.                                     |
| **`melt()`**            | Cambiar datos de formato wide a long.                                                |
| **Feature Engineering** | Crear o transformar variables para mejorar el análisis/modelado.                     |

---

## Mapa mental final

```bash
                         DATA SCIENCE
                              │
                              ▼
                       ADQUISICIÓN
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
            SQL              API              CSV
                              │
                              ▼
                            JSON
                              │
                              ▼
                           Pandas
                              │
                              ▼
                       DATA WRANGLING
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  Descubrimiento       Estructuración          Limpieza
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                       Enriquecimiento
                              │
                              ▼
                         Integración
                     merge() / concat()
                              │
                              ▼
                         Transformación
                              │
                              ▼
                          Validación
                              │
                              ▼
                     DATASET CONFIABLE
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
                EDA                        ML
                 │                         │
                 ▼                         ▼
              INSIGHTS                PREDICCIONES
```



