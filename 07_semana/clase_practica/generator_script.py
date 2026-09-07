import numpy as np
import pandas as pd

# Configuración inicial
RANDOM_SEED = 2026
N_RECORDS = 1500
rng = np.random.default_rng(RANDOM_SEED)

# Zonas adaptadas a la geografía local
zonas = rng.choice(
    ["Norte (Belgrano/Nuñez)", "Sur (Barracas/Avellaneda)", "Oeste (Flores/Liniers)", "Centro (Microcentro)"], 
    size=N_RECORDS, 
    p=[0.25, 0.15, 0.25, 0.35]
)
meses = rng.choice(["Ene", "Feb", "Mar", "Abr", "May", "Jun"], size=N_RECORDS)
ids = [f"BUE-{str(i).zfill(4)}" for i in range(1, N_RECORDS + 1)]

# Coordenadas geográficas centradas en CABA y AMBA
# Latitud: desde Gral Paz norte (-34.53) hasta Riachuelo (-34.65)
# Longitud: desde el Río (-58.35) hasta Gral Paz oeste (-58.53)
latitudes = rng.uniform(-34.65, -34.53, N_RECORDS)
longitudes = rng.uniform(-58.53, -58.35, N_RECORDS)

# Distancia en kilómetros puros (simulando trayectos cortos/medianos urbanos)
distancia_km = rng.exponential(scale=4.5, size=N_RECORDS).clip(0.8, 22.0)

# El tiempo en Buenos Aires depende fuertemente de la zona por el tráfico
# Penalización en minutos por congestión
factor_trafico = np.where(zonas == "Centro (Microcentro)", 25, 
                 np.where(zonas == "Sur (Barracas/Avellaneda)", 18, 5))

# Calculamos tiempo: a 20 km/h de promedio (3 min por km) + congestión + semáforos/ruido
tiempo_entrega_min = (distancia_km * 3.0) + factor_trafico + rng.normal(8, 4, N_RECORDS)
tiempo_entrega_min = tiempo_entrega_min.clip(10, 150).round(1)

# Estructura de costos en ARS: Bajada de bandera de $2500 + $950 por kilómetro
costo_envio_ars = (2500 + (distancia_km * 950) + rng.normal(0, 300, N_RECORDS)).clip(3000, None).round(0)

# La satisfacción (1 a 5 estrellas) cae si tardan más de 45 minutos
prob_base = 5.5 - (tiempo_entrega_min / 40) + rng.normal(0, 0.4, N_RECORDS)
satisfaccion_cliente = np.round(prob_base).clip(1, 5).astype(int)

# Armado del DataFrame
df_logistica = pd.DataFrame({
    "id_pedido": ids,
    "mes": meses,
    "zona": zonas,
    "latitud": latitudes,
    "longitud": longitudes,
    "distancia_km": distancia_km.round(2),
    "tiempo_entrega_min": tiempo_entrega_min,
    "costo_envio_ars": costo_envio_ars,
    "satisfaccion_cliente": satisfaccion_cliente
})

# Exportar para la clase
df_logistica.to_csv("dataset_logistica_caba.csv", index=False)
print("Dataset 'dataset_logistica_caba.csv' generado con éxito. Listo para la clase.")
