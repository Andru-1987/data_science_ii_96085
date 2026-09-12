import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600

barrios = np.array(["Norte", "Centro", "Sur", "Costa"])
barrio_mult = {"Norte": 1.25, "Centro": 1.0, "Sur": 0.75, "Costa": 1.6}
barrio = rng.choice(barrios, size=n, p=[0.25, 0.35, 0.25, 0.15])

tipos = np.array(["Casa", "Departamento", "PH"])
tipo = rng.choice(tipos, size=n, p=[0.35, 0.45, 0.20])
tipo_mult = {"Casa": 1.15, "Departamento": 1.0, "PH": 0.9}

# metros cuadrados: lognormal para generar asimetría positiva realista
m2 = rng.lognormal(mean=4.3, sigma=0.35, size=n)  # ~ 60-150 m2 con cola larga
m2 = np.clip(m2, 25, None)

# antigüedad: relación NO lineal (en U, no monótona) con el precio -> a propósito para Tip 1 y 8.3
antiguedad = rng.integers(0, 60, size=n)
efecto_antiguedad_frac = 0.00035 * (antiguedad - 25) ** 2  # forma de U, mínimo en ~25 años, como fracción del precio base

ambientes = np.clip(np.round(m2 / 28 + rng.normal(0, 0.4, n)), 1, 6)

precio_base = 900 * m2  # precio base por m2
ruido_precio = rng.lognormal(mean=0, sigma=0.15, size=n)

precio_usd = np.array([
    precio_base[i] * barrio_mult[barrio[i]] * tipo_mult[tipo[i]]
    * (1 + efecto_antiguedad_frac[i]) * ruido_precio[i]
    for i in range(n)
])

# un puñado de propiedades de lujo para reforzar la cola derecha (outliers reales)
idx_lujo = rng.choice(n, size=12, replace=False)
precio_usd[idx_lujo] *= rng.uniform(2.2, 3.5, size=12)

# impuestos anuales: fuertemente correlacionados con precio -> multicolinealidad (Tip 3)
impuestos_anuales_usd = precio_usd * 0.012 * (1 + rng.normal(0, 0.10, n))
impuestos_anuales_usd = np.clip(impuestos_anuales_usd, 200, None)

dias_en_mercado = rng.poisson(lam=45, size=n) + (antiguedad // 10)

df = pd.DataFrame({
    "id": range(1, n + 1),
    "barrio": barrio,
    "tipo_propiedad": tipo,
    "metros_cuadrados": m2.round(1),
    "ambientes": ambientes.astype(int),
    "antiguedad_anios": antiguedad,
    "precio_usd": precio_usd.round(0),
    "impuestos_anuales_usd": impuestos_anuales_usd.round(0),
    "dias_en_mercado": dias_en_mercado,
})

df.to_csv("propiedades.csv", index=False)
print(df.head(8).to_string())
print("\nShape:", df.shape)