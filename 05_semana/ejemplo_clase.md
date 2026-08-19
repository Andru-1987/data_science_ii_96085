# Data Wrangling II E-Commerce

Imaginemos que somos analistas de datos trabajando con el registro diario de transacciones de una tienda online. A lo largo de los ejemplos, iremos limpiando y preparando este mismo conjunto de datos para un futuro modelo de Machine Learning.

Primero, imaginemos nuestro conjunto de datos inicial crudo:

```python
import pandas as pd
import numpy as np

# Nuestro dataset crudo de ventas
data = {
    'id_transaccion': [1001, 1002, 1002, 1003, 1004, 1005],
    'cliente_zona': ['Norte', 'Sur', 'Sur', 'Este', 'Norte', 'Oeste'],
    'categoria': ['Electrónica', 'Hogar', 'Hogar', 'Ropa', 'Electrónica', 'Ropa'],
    'monto_usd': [250.0, 50.0, 50.0, 30.0, 120.0, 45.0],
    'estado_envio': ['Entregado', 'En tránsito', 'En tránsito', 'Entregado', 'Devuelto', 'Entregado']
}

df_ecommerce = pd.DataFrame(data)
print(df_ecommerce)

```

---

## **1. Manejo de Valores Duplicados**

*Ángulo: Limpieza de errores de sistema.*
A veces, por errores en la pasarela de pagos o en la red, una misma transacción se registra dos veces. Lo primero que debe hacer un Data Scientist es asegurar la integridad de la base.

```python
# Vemos que la transacción 1002 está duplicada exacta. La eliminamos.
df_ecommerce.drop_duplicates(inplace=True)

print("Dataset sin duplicados del sistema:")
print(df_ecommerce)

```

---

## **2. Transformación de Datos**

*Ángulo: Feature Engineering (Ingeniería de características).*
Para nuestro análisis financiero, el monto en USD puro no nos sirve si no calculamos los impuestos locales. Vamos a transformar los datos creando una nueva columna con el valor real facturado.

```python
# Agregamos el IVA (21%) al monto original
impuesto = 1.21
df_ecommerce['monto_facturado'] = df_ecommerce['monto_usd'] * impuesto

print("Datos con la nueva variable calculada:")
print(df_ecommerce[['id_transaccion', 'monto_usd', 'monto_facturado']])

```

---

## **3. Translación y Mapping**

*Ángulo: Preparación para algoritmos (Encoding).*
Los modelos de Machine Learning no entienden texto como "Entregado" o "En tránsito". Necesitamos mapear estos estados categóricos a valores numéricos (0, 1, 2) que el algoritmo sí pueda procesar.

```python
# Mapeo de estados logísticos a códigos numéricos
diccionario_estados = {
    'Devuelto': 0,
    'En tránsito': 1,
    'Entregado': 2
}

df_ecommerce['estado_codificado'] = df_ecommerce['estado_envio'].map(diccionario_estados)
print(df_ecommerce[['id_transaccion', 'estado_envio', 'estado_codificado']])

```

*(Nota: La translación con `shift()` aquí se usaría si tuviéramos las compras ordenadas por fecha para un mismo cliente y quisiéramos saber "cuántos días pasaron desde su compra anterior").*

---

## **4. Filtros y Selección de Datos**

*Ángulo: Segmentación de clientes de alto valor.*
Desde el equipo de marketing nos piden aislar únicamente las compras exitosas (entregadas) que superen los 100 dólares para enviarles un cupón VIP.

```python
# Máscara booleana con múltiples condiciones
mascara_vip = (df_ecommerce['estado_envio'] == 'Entregado') & (df_ecommerce['monto_usd'] > 100)

df_vip = df_ecommerce[mascara_vip]
print("Transacciones VIP:")
print(df_vip)

```

---

## **5. Índices Jerárquicos (MultiIndex)**

*Ángulo: Análisis multidimensional.*
Queremos estructurar nuestra base de datos para acceder rápidamente a la información basándonos en la Zona del cliente y luego en la Categoría del producto que compraron.

```python
# Establecemos dos columnas como nuestro nuevo índice jerárquico
df_indexado = df_ecommerce.set_index(['cliente_zona', 'categoria'])

# Ordenamos el índice para mejor visualización y rendimiento
df_indexado.sort_index(inplace=True)

print("Datos indexados por Zona y Categoría:")
print(df_indexado)

```

---

## **6. GroupBy, Apply y Pivot**

*Ángulo: Reporte ejecutivo de ventas.*

El gerente de ventas quiere ver de un vistazo el ticket promedio (media de USD) vendido por cada Zona geolocalizada, separando por Categoría. `pivot_table` cruza esta información perfectamente.

```python
# Filas: Zona. Columnas: Categoría. Valores: Promedio del monto_usd.
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

```

---

## **7. Melt y Reestructuración**

*Ángulo: Preparación de datos para graficar en Seaborn/Tableau.*

A veces, herramientas de visualización necesitan que los datos de un reporte (como el que acabamos de crear en el paso 6) estén "desdoblados" o en formato largo (long format) para poder graficar correctamente los ejes X e Y con colores por categoría.

```python
# Reseteamos el índice del reporte anterior para que 'cliente_zona' vuelva a ser una columna
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

```
