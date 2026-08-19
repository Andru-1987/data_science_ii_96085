import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL:str = os.getenv("NEON_PG_DATABASE")

engine = create_engine(DATABASE_URL)

# 3. Definir la consulta SQL
# Vamos a hacer un JOIN entre Customers y Orders como ejemplo
query = """
    SELECT * FROM ecommerce_transacciones
"""

# 4. Conectar y extraer los datos a un DataFrame
# Usamos un bloque 'with' para asegurar que la conexión se cierre correctamente
with engine.connect() as connection:
    # Pandas ejecuta la consulta y convierte el resultado en un DataFrame
    df_ecommerce = pd.read_sql_query(sql=text(query), con=connection)


print("Frame directamente desde Neon DB")
print(df_ecommerce.head())


print("Remover duplicados")
df_ecommerce.drop_duplicates(inplace=True)

print("Dataset sin duplicados del sistema:")
print(df_ecommerce)


print("Transformacion de datos")
impuesto = 1.21
df_ecommerce['monto_facturado'] = df_ecommerce['monto_usd'] * impuesto

print("Datos con la nueva variable calculada:")
print(df_ecommerce[['id_transaccion', 'monto_usd', 'monto_facturado']])



print("Mapeo de estados logísticos a códigos numéricos")

diccionario_estados = {
    'Devuelto': 0,
    'En tránsito': 1,
    'Entregado': 2
}

df_ecommerce['estado_codificado'] = df_ecommerce['estado_envio'].map(diccionario_estados)
print(df_ecommerce[['id_transaccion', 'estado_envio', 'estado_codificado']])

print("Usando Shift")

df_ecommerce['fecha_transaccion'] = pd.to_datetime(df_ecommerce['fecha_transaccion'])

# 3. Ordenamos cronológicamente por Zona y luego por Fecha
df_ecommerce = df_ecommerce.sort_values(by=['cliente_zona', 'fecha_transaccion'])

# 4. Usamos shift(1) agrupando por zona para traer la fecha de la fila anterior a la actual
df_ecommerce['fecha_compra_anterior'] = df_ecommerce.groupby('cliente_zona')['fecha_transaccion'].shift(1)

# 5. Calculamos la diferencia en días
df_ecommerce['dias_desde_ultima_venta'] = (df_ecommerce['fecha_transaccion'] - df_ecommerce['fecha_compra_anterior']).dt.days

# Mostramos el resultado enfocado en las columnas relevantes, ordenado por zona
print(df_ecommerce[['id_transaccion', 'cliente_zona', 'fecha_transaccion', 'fecha_compra_anterior', 'dias_desde_ultima_venta']])


print("Máscara booleana con múltiples condiciones")
mascara_vip = (df_ecommerce['estado_envio'] == 'Entregado') & (df_ecommerce['monto_usd'] > 100)

df_vip = df_ecommerce[mascara_vip]
print("Transacciones VIP:")
print(df_vip)

print("Establecemos dos columnas como nuestro nuevo índice jerárquico")
df_indexado = df_ecommerce.set_index(['cliente_zona', 'categoria'])

# Ordenamos el índice para mejor visualización y rendimiento
df_indexado.sort_index(inplace=True)

print("Datos indexados por Zona y Categoría:")
print(df_indexado)


print("Filas: Zona. Columnas: Categoría. Valores: Promedio del monto_usd.")
reporte_ventas = pd.pivot_table(
    df_ecommerce, 
    values='monto_usd', 
    index='cliente_zona', 
    columns='categoria', 
    aggfunc='mean',
    fill_value=0 # Llenamos con 0 si no hubo ventas en ese cruce
)

print("Reporte Ejecutivo (Ticket Promedio):")
print(reporte_ventas)

print("Reseteamos el índice del reporte anterior para que 'cliente_zona' vuelva a ser una columna")
reporte_plano = reporte_ventas.reset_index()

# Usamos Melt para desdoblar las categorías de productos
df_largo_visualizacion = pd.melt(
    reporte_plano, 
    id_vars=['cliente_zona'], 
    value_vars=['Electrónica', 'Hogar', 'Ropa'], 
    var_name='categoria_producto', 
    value_name='ticket_promedio'
)

print("Datos reestructurados para visualización:")
print(df_largo_visualizacion)