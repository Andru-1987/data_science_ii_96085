## 1. MongoDB Atlas

```python
from pymongo import MongoClient

# Conexion a MongoDB Atlas
# Reemplaza <username>, <password>, <cluster> y <database> con tus datos
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>"

# Opcion 1: Conexion basica
client = MongoClient(MONGO_URI)
db = client['nombre_base_datos']

# Opcion 2: Conexion con parametros adicionales
client = MongoClient(
    MONGO_URI,
    maxPoolSize=50,
    minPoolSize=10,
    retryWrites=True,
    w='majority'
)

# Ejemplo de uso
db = client['mi_base_datos']
coleccion = db['mi_coleccion']

# Obtener documentos
resultados = coleccion.find()
for doc in resultados:
    print(doc)

# Insercion de ejemplo
nuevo_documento = {
    "nombre": "Juan",
    "edad": 30,
    "ciudad": "Madrid"
}
coleccion.insert_one(nuevo_documento)
```

## 2. Supabase (PostgreSQL)

```python
import os
from supabase import create_client, Client
import pandas as pd
from sqlalchemy import create_engine

# Configuracion Supabase
SUPABASE_URL = "https://<tu-proyecto>.supabase.co"
SUPABASE_KEY = "<tu-api-key-publica>"

# Opcion 1: Usando el cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Obtener datos de una tabla
response = supabase.table('empleados').select("*").execute()
data = response.data

# Convertir a DataFrame
df = pd.DataFrame(data)
print(df.head())

# Insertar datos
nuevo_empleado = {
    "nombre": "Ana",
    "apellido": "Garcia",
    "departamento": "Ventas"
}
supabase.table('empleados').insert(nuevo_empleado).execute()

# Actualizar datos
supabase.table('empleados').update({
    "departamento": "Marketing"
}).eq('nombre', 'Ana').execute()

# Eliminar datos
supabase.table('empleados').delete().eq('nombre', 'Ana').execute()

# Opcion 2: Usando SQLAlchemy
SUPABASE_DB_URL = f"postgresql://postgres:<tu-password>@db.<tu-proyecto>.supabase.co:5432/postgres"

engine = create_engine(SUPABASE_DB_URL)

df = pd.read_sql('SELECT * FROM empleados WHERE departamento = "Ventas"', engine)
print(df.head())

# Opcion 3: Usando psycopg2
import psycopg2

conn = psycopg2.connect(
    host="db.<tu-proyecto>.supabase.co",
    database="postgres",
    user="postgres",
    password="<tu-password>",
    port=5432
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM empleados")
rows = cursor.fetchall()
```

## Instalacion de dependencias

```bash
pip install pymongo dnspython
pip install supabase postgrest-py psycopg2-binary sqlalchemy
pip install pandas
```

## Variables de entorno

```python
# .env file
MONGO_URI=mongodb+srv://usuario:password@cluster.mongodb.net/database
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu-api-key
SUPABASE_DB_PASSWORD=tu-password

# En tu codigo
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)

# Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
```

## Consideraciones importantes

MongoDB Atlas:
- Usa mongodb+srv:// para el connection string de Atlas
- Asegurate de que tu IP este en la whitelist de Atlas
- Las contrasenas deben estar correctamente encoded para URL

Supabase:
- El API Key debe ser la publica (no la service role key)
- Las tablas deben tener RLS (Row Level Security) configurado
- Para conexiones SQL directas, necesitas la contrasena de la base de datos