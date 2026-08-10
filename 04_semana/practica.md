# APIs y Data Wrangling con Pandas

En Data Science, la obtención de los datos es solo el punto de partida. El verdadero desafío radica en transformar datos crudos y heterogéneos en información confiable y estructurada para su posterior análisis o uso en modelos de Machine Learning.

## 1. APIs y JSON: Conectando Sistemas
Una **API (Application Programming Interface)** actúa como un puente que permite a dos sistemas comunicarse. Para un Científico de Datos, las APIs son fuentes vitales de información externa.

* **El ciclo básico:** `Cliente ➔ Request (GET) ➔ API ➔ Response (Estado 200) ➔ Cliente`
* **JSON:** Las APIs devuelven la información en formato JSON (JavaScript Object Notation). Es un formato semi-estructurado de `clave ➔ valor` que debe transformarse a un formato tabular (DataFrame) para ser analizado.

## 2. Data Wrangling (Las 6 Etapas)
Es el proceso sistemático de limpieza, transformación y unificación de datos desordenados.

1. **Descubrimiento:** Explorar los datos (`info()`, `describe()`, `shape`).
2. **Estructuración:** Homogeneizar formatos y tipos de datos.
3. **Limpieza:** Tratar nulos (`fillna()`), remover duplicados (`drop_duplicates()`) y normalizar texto (`str.lower()`).
4. **Enriquecimiento:** Cruzar información (`merge()`, `concat()`).
5. **Validación:** Comprobar mediante reglas lógicas (`assert`).
6. **Publicación:** Exportar el dataset limpio.

## 3. Operaciones Clave en Pandas
* **Integración:**
  * `merge()`: Combina tablas mediante claves en común (Similar a `LEFT JOIN` en SQL).
  * `concat()`: Apila tablas con la misma estructura (una debajo de la otra).
* **Agregación:** `groupby()` permite agrupar datos por categorías y calcular métricas (sumas, promedios).
* **Transformación (Feature Engineering):** Crear nuevas variables a partir de las existentes (ej. calcular un precio con IVA).

## Resumen de Conceptos Clave

| Concepto | ¿Qué problema resuelve? |
|----------|-------------------------|
| **`df.info()` / `df.shape`** | Permite conocer el volumen de datos (filas/columnas) y los tipos de variables. |
| **`df.isna().sum()`** | Detecta qué columnas tienen datos faltantes para poder imputarlos o eliminarlos. |
| **`df.fillna(mediana)`** | Imputa valores nulos. La mediana es preferible a la media porque no se ve afectada por valores extremos (outliers). |
| **`df.str.strip().str.lower()`** | Normaliza texto. Vital antes de un `groupby` para evitar que "Cordoba", "CORDOBA" y " cordoba" se cuenten como ciudades distintas. |
| **`merge()` vs `concat()`** | `merge()` relaciona datos por un ID común. `concat()` simplemente apila filas. |

## Mapa Mental del Flujo de Datos

```text
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
  Descubrimiento       Estructuración           Limpieza
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

```

---

### 2. Archivo Python: Ejercitación Completa
Copia el siguiente código y guárdalo como `ejercicios_wrangling.py`. Asegúrate de tener instaladas las librerías (`pip install pandas requests`). Para que los ejercicios de API funcionen, he utilizado una API pública gratuita real (JSONPlaceholder).

```python
import pandas as pd
import requests

# ---------------------------------------------------------
# Ejercicio 1 — Explorar un dataset
# ---------------------------------------------------------
print("--- Ejercicio 1: Exploración ---")
data = {
    "nombre": ["Juan", "Maria", "Pedro", "Juan"],
    "edad": [25, 32, None, 25],
    "ciudad": ["Buenos Aires", "CORDOBA", "cordoba", "Buenos Aires"],
    "ventas": [1000, 2000, 1500, 1000]
}
df = pd.DataFrame(data)

print("Filas y columnas (shape):", df.shape)
print("\nInformación del DataFrame:")
df.info()
print("\nValores faltantes por columna:")
print(df.isna().sum())
print("\nCantidad de duplicados:", df.duplicated().sum())
print("\n")

# ---------------------------------------------------------
# Ejercicio 2 — Limpiar valores de texto
# ---------------------------------------------------------
print("--- Ejercicio 2: Normalización de texto ---")
df["ciudad"] = df["ciudad"].str.strip().str.lower()
print("Ciudades normalizadas:")
print(df["ciudad"].value_counts())
print("\n")

# ---------------------------------------------------------
# Ejercicio 3 — Valores faltantes
# ---------------------------------------------------------
print("--- Ejercicio 3: Imputación de nulos ---")
mediana = df["edad"].median()
df["edad"] = df["edad"].fillna(mediana)
print("Valores faltantes en 'edad' después de fillna:", df["edad"].isna().sum())
print(df[["nombre", "edad"]])
print("\n")

# ---------------------------------------------------------
# Ejercicio 4 — Detectar duplicados
# ---------------------------------------------------------
print("--- Ejercicio 4: Eliminación de duplicados ---")
df = df.drop_duplicates()
print("Duplicados restantes:", df.duplicated().sum())
print(df)
print("\n")

# ---------------------------------------------------------
# Ejercicio 5 — Crear una nueva variable
# ---------------------------------------------------------
print("--- Ejercicio 5: Feature Engineering ---")
df["ventas_con_iva"] = df["ventas"] * 1.21
print(df[["ventas", "ventas_con_iva"]])
print("\n")

# ---------------------------------------------------------
# Ejercicio 6 — Agrupar información
# ---------------------------------------------------------
print("--- Ejercicio 6: Groupby ---")
agrupado = df.groupby("ciudad").agg(
    ventas_total=("ventas", "sum"),
    venta_promedio=("ventas", "mean"),
    clientes=("nombre", "count")
)
print(agrupado)
print("\n")

# ---------------------------------------------------------
# Ejercicio 7 — merge()
# ---------------------------------------------------------
print("--- Ejercicio 7: Merge (Left Join) ---")
clientes = pd.DataFrame({
    "cliente_id": [1, 2, 3],
    "nombre": ["Juan", "Maria", "Pedro"]
})
compras = pd.DataFrame({
    "cliente_id": [1, 1, 2],
    "producto": ["Laptop", "Mouse", "Teclado"],
    "importe": [1000, 50, 80]
})

resultado_merge = clientes.merge(compras, on="cliente_id", how="left")
print(resultado_merge)
# Nota: Pedro aparece con NaN en producto e importe porque es un LEFT JOIN
print("\n")

# ---------------------------------------------------------
# Ejercicio 8 — concat()
# ---------------------------------------------------------
print("--- Ejercicio 8: Concat (Apilar) ---")
enero = pd.DataFrame({"producto": ["Laptop", "Mouse"], "ventas": [1000, 200]})
febrero = pd.DataFrame({"producto": ["Laptop", "Mouse"], "ventas": [1200, 250]})

ventas_totales = pd.concat([enero, febrero], ignore_index=True)
print(ventas_totales)
print("\n")

# ---------------------------------------------------------
# Ejercicio 9 y 10 integrados — API + Pipeline Data Wrangling
# ---------------------------------------------------------
print("--- Ejercicio Integrador: API + Pipeline Completo ---")
# Usamos una API pública de prueba para que el código funcione
url_api = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url_api)

if response.status_code == 200:
    print("Conexión exitosa a la API (Status: 200)\n")
    
    # 1. Convertir a JSON y crear DataFrame
    data_api = response.json()
    df_api = pd.DataFrame(data_api)
    
    # 2. Descubrimiento
    print("Shape original:", df_api.shape)
    
    # 3. Limpieza y Extracción (La API anida la ciudad dentro de un dict 'address')
    # Extraemos la ciudad del diccionario para normalizarla
    df_api["ciudad"] = df_api["address"].apply(lambda x: x.get("city") if isinstance(x, dict) else None)
    df_api["ciudad"] = df_api["ciudad"].str.strip().str.lower()
    
    # 4. Feature Engineering (Simulamos una columna de ventas aleatorias)
    import numpy as np
    np.random.seed(42)
    df_api["ventas"] = np.random.randint(100, 1000, size=len(df_api))
    
    # 5. Validación
    assert df_api["ciudad"].notna().all(), "Error: Hay valores nulos en ciudad"
    print("Validación superada: No hay nulos en 'ciudad'.")
    
    # 6. Agrupación (Agrupamos por si casualmente hubiera ciudades repetidas, aunque aquí son únicas)
    agrupacion_api = df_api.groupby("ciudad").agg(
        ventas_totales=("ventas", "sum")
    ).head(3)
    
    print("\nMuestra del dataset final procesado (Nombre, Ciudad, Ventas):")
    print(df_api[["name", "ciudad", "ventas"]].head())
    
    print("\nMuestra de métricas agrupadas:")
    print(agrupacion_api)

else:
    print("Error al conectar con la API. Código:", response.status_code)

```
