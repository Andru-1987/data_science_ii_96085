import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()

# 1. Configurar la cadena de conexión (Reemplaza con tus credenciales de Supabase)
USER = os.getenv("user")
PASSWORD = os.getenv("DB_PG_SUPABASE")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")
# Construct the SQLAlchemy connection string

DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
# 2. Crear el motor de conexión (Engine)
engine = create_engine(DATABASE_URL)

# 3. Definir la consulta SQL
# Vamos a hacer un JOIN entre Customers y Orders como ejemplo
query = """
    SELECT 
        c.CustomerName, 
        c.City, 
        c.Country, 
        o.OrderDate, 
        o.Amount
    FROM Customers c
    INNER JOIN Orders o ON c.CustomerID = o.CustomerID
    WHERE o.Amount > 100.00
"""

# 4. Conectar y extraer los datos a un DataFrame
# Usamos un bloque 'with' para asegurar que la conexión se cierre correctamente
with engine.connect() as connection:
    # Pandas ejecuta la consulta y convierte el resultado en un DataFrame
    df = pd.read_sql_query(sql=text(query), con=connection)

# 5. Visualizar los resultados
print("Datos extraídos de Supabase:")
print(df)

# Ejemplo de operación con Pandas: Agrupar por país y sumar el monto
print("\nTotal de ventas por país:")
ventas_por_pais = df.groupby('country')['amount'].sum().reset_index()
print(ventas_por_pais)