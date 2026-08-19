## **Transformación de Datos (Data Transformation)**

* La transformación de datos implica cambiar la estructura, formato o valor de los datos para alinearlos con el tipo de análisis a realizar.
* Incluye operaciones fundamentales como el escalado de valores, filtrado, selección, agregación, codificación de variables categóricas y mapeo de valores.

## **Manejo de Valores Duplicados**

* Es crucial identificar y manejar datos duplicados, ya que pueden sesgar los resultados del análisis y llevar a conclusiones incorrectas.
* La función **`duplicated()`** evalúa el DataFrame y devuelve una serie booleana (True/False) indicando si una fila es idéntica a otra.
* La función **`drop_duplicates()`** elimina las filas repetidas, permitiendo ajustar su comportamiento mediante parámetros como `subset` (columnas específicas), `keep` (qué aparición conservar) e `inplace` (modificar el DataFrame original).

## **Índices Jerárquicos (MultiIndex)**

* Un MultiIndex permite crear índices con múltiples niveles para manejar bases de datos complejas con varias dimensiones categóricas.
* Esta herramienta facilita la agrupación y organización detallada de los datos, permitiendo estructurarlos de manera jerárquica (por ejemplo, agrupar primero por País y luego por Ciudad).

## **Translación y Mapping**

* **Translación:** Utiliza la función **`shift()`** para desplazar los valores de una columna hacia arriba o hacia abajo.
* Esta técnica de desplazamiento es fundamental para operaciones secuenciales, como calcular variaciones entre períodos en series temporales.
* **Mapping:** Emplea la función **`map()`** para transformar y reemplazar rápidamente los valores de una columna utilizando un diccionario fijo o una función con lógica personalizada.

## **Filtros y Selección de Datos**

* La selección de datos basada en condiciones se realiza mediante **máscaras booleanas**.
* Una máscara booleana es un conjunto de valores indicando si cada fila cumple o no con un criterio específico para filtrar la información relevante.

## **GroupBy, Apply y Pivot**

* **Apply:** La función **`apply()`** sirve para ejecutar funciones personalizadas sobre subconjuntos de datos previamente agrupados mediante `groupby()`.
* **Pivot:** La función **`pivot_table()`** reorganiza los datos en una tabla dinámica para analizar información agregada.
* Con `pivot_table()` puedes definir índices, columnas y aplicar múltiples funciones de agregación estadísticas de forma simultánea (como `sum`, `mean` y `count`).

## **Melt y Reestructuración**

* La función **`melt()`** se encarga de transformar la estructura de un DataFrame de un formato amplio (wide) a un formato largo (long).
* Requiere configurar parámetros clave como `id_vars` (identificadores fijos), `value_vars` (columnas a derretir), `var_name` y `value_name` para reorganizar eficientemente los datos de cara a un análisis o visualización.