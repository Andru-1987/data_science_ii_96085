#  Gestión Avanzada de BD con SQL y Python

##  Objetivos de la Unidad

Al finalizar esta unidad, el estudiante será capaz de:

1. **Dominar SQL Avanzado:** Ejecutar transacciones (TCL), agrupaciones condicionales y uniones entre múltiples tablas.
2. **Integrar Python y Bases de Datos:** Conectar bases de datos relacionales utilizando `SQLAlchemy`.
3. **Manejar APIs y Datos No Estructurados:** Extraer, aplanar y analizar datos de archivos JSON y APIs utilizando `requests` y `Pandas`.

---

## Módulo 1: SQL Avanzado

### 1.1 Filtrado Preciso (WHERE y Operadores Lógicos)

Permite extraer registros que cumplen condiciones estrictas utilizando combinaciones de operadores.

* **WHERE:** Filtra la tabla base.
* **AND / OR / NOT:** Operadores lógicos para anidar múltiples condiciones.

### 1.2 Uniones de Tablas (JOINs)

Cruzan información entre dos o más tablas a través de claves relacionales.

* **INNER JOIN:** Retorna solo las filas con coincidencias en *ambas* tablas.
* **LEFT JOIN:** Retorna *toda* la tabla izquierda, más coincidencias de la derecha (rellena con nulos si no hay).
* **RIGHT JOIN:** Retorna *toda* la tabla derecha, más coincidencias de la izquierda.
* **FULL JOIN:** Combina LEFT y RIGHT; retorna todo si hay coincidencia en *cualquiera* de las tablas.

### 1.3 Agrupación y Condicionales (GROUP BY y HAVING)

Diseñados para análisis agregado y métricas.

* **GROUP BY:** Colapsa filas con valores idénticos en grupos. Se utiliza obligatoriamente con funciones agregadas (`COUNT`, `SUM`, `AVG`, `MAX`, `MIN`).
* **HAVING:** Es el "WHERE de los grupos". Filtra los resultados *después* de que la agregación se ha calculado.

### 1.4 Transacciones Seguras (TCL)

Asegura la integridad de los datos en bases de datos de producción (ACID).

* **BEGIN:** Inicia el bloque transaccional.
* **COMMIT:** Confirma y guarda los cambios de forma permanente.
* **ROLLBACK:** Revierte los cambios si ocurre un error, devolviendo la base de datos a su estado anterior.
* **SAVEPOINT:** Crea un punto de retorno temporal dentro de una transacción en curso.

---

## Módulo 2: Ecosistema Python para Bases de Datos

### 2.1 Conectores SQL en Python

* **SQLite3:** Librería nativa de Python. Ideal para bases de datos locales ligeras. Se conecta directo al archivo `.db`.
* **SQLAlchemy:** ORM (Object-Relational Mapping) avanzado. Crea un "motor" (`engine`) flexible que permite conectar Python con múltiples gestores (PostgreSQL, MySQL, SQL Server) usando una sintaxis unificada.

### 2.2 Pandas y Archivos JSON

El JSON es el estándar web de intercambio de datos (clave-valor, semi-estructurado).

* `pd.read_json()`: Transforma automáticamente un archivo local o URL en un DataFrame.
* `pd.json_normalize()`: Esencial para aplanar (flatten) estructuras JSON complejas o anidadas (diccionarios dentro de diccionarios).

---

# Explicación Técnica: De APIs a Pandas DataFrame

Para consumir una API pública y llevarla a Pandas, seguimos un flujo de trabajo estándar de 3 pasos:

1. **Petición (Request):** Usamos la librería `requests` con el método `get()` para ir a la URL de la API y pedir los datos.
2. **Validación y Extracción:** Comprobamos que `response.status_code == 200` (respuesta exitosa) y extraemos la carga útil usando `.json()`. Esto convierte el texto de la API en diccionarios y listas nativas de Python.
3. **Ingesta (DataFrame):** Pasamos esa lista/diccionario a `pd.DataFrame()` o usamos `pd.json_normalize()` si los datos están anidados.

A continuación, 3 ejercicios prácticos con 3 APIs distintas:

### Ejercicio 1: API Sencilla (Lista plana de diccionarios)

**API a utilizar:** JSONPlaceholder (Usuarios)
*Escenario:* Consumir una lista de usuarios de prueba y transformarla en un DataFrame básico.

```python
import requests
import pandas as pd

# 1. Realizar la solicitud
url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url)

# 2. Validar y procesar
if response.status_code == 200:
    data = response.json() # Retorna una lista de diccionarios
    
    # 3. Convertir a DataFrame
    df_users = pd.DataFrame(data)
    
    # Visualizar las columnas principales
    print(df_users[['id', 'name', 'email', 'phone']].head())
else:
    print(f"Error en la conexión: {response.status_code}")

```

### Ejercicio 2: Api Keys

**API a utilizar:** REST Countries
> Version actual con API_KEY: [REST COUNTRIES](https://restcountries.com/sign-up)

[archivo de ejemplo](./python_files/api_key.py)


#### Aplanamiento de JSON Anidado

**Objetivo:** Utilizar `json_normalize` para manejar estructuras de datos complejas.
**Consigna:** Realiza una petición a JSONPlaceholder (`[https://jsonplaceholder.typicode.com/users](https://jsonplaceholder.typicode.com/users)`). Como notarás, la clave `address` contiene un diccionario con detalles como calle y ciudad, y dentro de ella, `geo` contiene otro diccionario. Utiliza `pd.json_normalize` para crear un DataFrame donde todas estas claves queden aplanadas. Muestra las columnas referentes al nombre, ciudad y coordenadas.
**Solución esperada:**

```python
import requests
import pandas as pd

url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    # Aplanado del JSON
    df_users = pd.json_normalize(data)
    
    # Selección de columnas generadas por el aplanado
    columnas = ['name', 'address.city', 'address.geo.lat', 'address.geo.lng']
    print(df_users[columnas].head())
else:
    print("Error:", response.status_code)

```