import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

url: str = "https://api.restcountries.com/countries/v5?region=Europe"
headers: dict = {'Authorization': f'Bearer {os.getenv("REST_API_KEY")}'}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    json_response = response.json()
    paises_data = json_response['data']['objects']
    
    # 1. Creamos una lista vacía para guardar la información limpia
    datos_limpios = []
    
    # 2. Recorremos cada país para extraer solo el texto que nos sirve
    for pais in paises_data:
        
        # Extraemos el nombre común
        nombre = pais.get('names', {}).get('common', 'Desconocido')
        
        # Extraemos la población
        poblacion = pais.get('population', 0)
        
        # Extraemos el nombre de la capital (puede haber más de una, las unimos con comas)
        capitales = pais.get('capitals', [])
        capital = ", ".join([c.get('name', '') for c in capitales]) if capitales else "Sin capital"
        
        # Extraemos el nombre de la moneda (puede haber más de una)
        monedas = pais.get('currencies', [])
        moneda = ", ".join([m.get('name', '') for m in monedas]) if monedas else "Sin moneda"
        
        # Agregamos este país a nuestra lista limpia
        datos_limpios.append({
            'Nombre': nombre,
            'Capital': capital,
            'Moneda': moneda,
            'Poblacion': poblacion
        })
        
    # 3. Ahora sí, convertimos la lista limpia directamente a un DataFrame
    df_final = pd.DataFrame(datos_limpios)
    
    # 4. Mostramos el resultado impecable
    print(df_final.head())
        
else:
    print(f"Error: {response.status_code} - {response.text}")