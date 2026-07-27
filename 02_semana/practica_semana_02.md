### 1. Importación de Librerías

```python
import pandas as pd
from google.colab import files
import io
```

### 2. Subir archivo desde la PC al entorno de Google Colab

```python
uploaded = files.upload()   # Permite subir archivos desde la pc local al entorno
import io                   # Permite manejar datos en forma de flujo

df2 = pd.read_csv(io.BytesIO(uploaded['nombre_archivo.csv']), sep=",")
df2.head()
df2.shape
```

### 3. Lectura de datos desde Google Colab

```python
# Configuramos el entorno de Python en Google Colab
from google.colab import drive
import os

drive.mount("/content/drive")
print(os.getcwd())

os.chdir("/content/drive/My Drive/")
print(os.getcwd())

file_url = "#usar_alguna_fuente.csv"

df2 = pd.read_csv(file_url, sep=",")
df2.info()
```

### 4. Lectura de un archivo RAW desde GitHub (.csv)

```python
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
df = pd.read_csv(url)
df
```

### 5. Lectura de un archivo RAW desde GitHub (.json)

```python
df = pd.read_json('http://raw.githubusercontent.com/BindiChen/machine-learning/master/data-analysis/027-pandas-convert-json/data/simple.json')
df.head(10)
```

### 6. Extracción de Datos desde APIs

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# API de astronautas
response = requests.get("http://api.open-notify.org/astros.json")
print(response.status_code)

data = response.json()
data

df = pd.DataFrame(data["people"])
df.head()

# API de Bitcoin
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Precio de Bitcoin (USD):", data["bitcoin"]["usd"])
else:
    print("Error:", response.status_code)

# Obtener datos actuales en DataFrame
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame([data["bitcoin"]])  # Convertir a DataFrame
    print("DataFrame del precio actual:")
    print(df)
else:
    print("Error:", response.status_code)

# Obtener datos históricos (últimos 30 días)
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    prices = data["prices"]

    # Crear DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convertir timestamp a fecha
    df = df.set_index("date").drop("timestamp", axis=1)  # Fecha como índice

    print("DataFrame histórico:")
    print(df.head())

    # Gráfico de línea
    df.plot(title="Precio de Bitcoin (Últimos 30 días)")
    plt.show()
else:
    print("Error:", response.status_code)
```

### 7. Web Scraping

```python
import requests
from bs4 import BeautifulSoup

# Obtener el contenido HTML de una página web
response = requests.get('https://www.coderhouse.com/ar/')
soup = BeautifulSoup(response.text, 'html.parser')

# Extraer un elemento específico (por ejemplo, el título)
titulo = soup.find('h1').text
print(titulo)
```

### 8. Scraping buscando una tabla en la página

```python
from io import StringIO
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL de la página
url = "https://es.wikipedia.org/wiki/Anexo:Tabla_estad%C3%ADstica_de_la_Copa_Mundial_de_F%C3%BAtbol"

# Encabezados para evitar bloqueos
headers = {
    "User-Agent": "Mozilla/5.0"
}

# Solicitud HTTP
response = requests.get(url, headers=headers)

# Parsear el HTML
soup = BeautifulSoup(response.text, "html.parser")

# Buscar todas las tablas de Wikipedia
tables = soup.find_all("table", class_="wikitable")

print(f"Se encontraron {len(tables)} tablas.")

# Seleccionar la primera tabla
table = tables[0]

# Convertir la tabla HTML a DataFrame
df = pd.read_html(StringIO(str(table)))[0]

# Mostrar las primeras filas
df.head()
df
```

### 9. Conexión a bases de datos con Python
> Recomendacion caso de uso en la nube en el anexo [ANEXO](bases_nube.md)

```python
from sqlalchemy import create_engine

# Crear conexión a la base de datos
engine = create_engine('mysql+pymysql://usuario:contraseña@localhost/nombre_base_datos')

# Ejecutar una consulta SQL y obtener el resultado como un DataFrame
df = pd.read_sql('SELECT * FROM empleados', engine)
df.head()
```

### 10. Trabajar con bases de datos no relacionales (NoSQL)
> Recomendacion caso de uso en la nube en el anexo [ANEXO](bases_nube.md)

```python
from pymongo import MongoClient

# Conectar a MongoDB
client = MongoClient('localhost', 27017)
db = client['nombre_base_datos']

# Obtener todos los documentos de una colección
result = db.coleccion.find()
for doc in result:
    print(doc)
```

### 11. Lectura de archivos planos

```python
# CSV
df = pd.read_csv('archivo.csv', sep=',')

# TSV
df = pd.read_csv('archivo.txt', delimiter='\t')

# TXT con separadores personalizados
df = pd.read_csv('archivo.txt', delimiter=';')  # ejemplo con punto y coma
```

### 12. Lectura de Archivos Sin Headers

```python
# Cargar un CSV de GitHub
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
df = pd.read_csv(url, sep=',', header=None)
df.head()

# Renombrar las columnas
df = df.rename(columns={
    0: "NEMB",   # Numero de veces embarazada
    1: "GLU",    # Concentracion de plasma de glucosa
    2: "PART",   # Presion arterial diastolica en mm Hg
    3: "GROS",   # Grosor de piel en triceps en mm
    4: "HUR",    # 2-Hour serum insulin en mu U/ml
    5: "BMI",    # BMI (peso kg/(altura en m)^2)
    6: "FPRED",  # Funcion de prediccion de Diabetes
    7: "AGE",    # Edad (años)
    8: "CLASS"   # Variable de clase (0 or 1)
})
df.head()
```

### 13. Uso del Formato Pickle

```python
# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'nombre': ['Juan', 'Ana', 'Luis'],
    'edad': [25, 30, 22],
    'ciudad': ['Buenos Aires', 'Mendoza', 'Cordoba']
})
df

# Guardar el DataFrame en un archivo Pickle
df.to_pickle('dataframe.pkl')

# Cargar el DataFrame desde el archivo Pickle
df_loaded = pd.read_pickle('dataframe.pkl')
df_loaded
```