# Guía Práctica en Python: Storytelling, GIS y Taxonomía del Análisis Estadístico

Todo el código de esta guía fue ejecutado y verificado antes de entregarlo. Los datasets son sintéticos (sandbox), generados dentro del propio script, así que se puede correr todo sin depender de archivos externos.

Librerías necesarias:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium geopandas shapely plotly
```

La Parte 3 usa `dataset_propiedades.csv` (150 propiedades), que se entrega junto con esta guía. Colocalo en la misma carpeta desde donde corras los scripts.

---

## Parte 1 — Storytelling Aplicado a Data Science

La idea de esta sección es mostrar, con el **mismo tipo de decisión de negocio**, cómo cambia el código según el formato narrativo elegido: reporte estático, dashboard interactivo o scrollytelling.

### 1.1 Reporte Estático

**Consigna:** sos analista en una cadena de retail y tenés que explicarle a la gerencia por qué cayeron las ventas en la zona Norte, en una reunión donde vos mismo vas a estar presentando el gráfico.

**Idea pedagógica clave:** en un reporte estático el gráfico no se entrega solo — es apoyo de una narrativa que el presentador cuenta en voz alta. Por eso el código no se limita a graficar: **anota directamente sobre la imagen** la causa raíz y resalta con color el punto de conflicto (estructura inicio → nudo → desenlace).

```python
import matplotlib.pyplot as plt
import pandas as pd

# --- Dataset sandbox: ventas mensuales por zona (en miles de $) ---
df_ventas_zona = pd.DataFrame({
    "zona": ["Norte", "Sur", "Este", "Oeste", "Centro"],
    "ventas_mes_actual": [180, 310, 295, 260, 340],
    "ventas_mes_anterior": [340, 300, 280, 255, 330],
})
df_ventas_zona["variacion_pct"] = (
    (df_ventas_zona["ventas_mes_actual"] - df_ventas_zona["ventas_mes_anterior"])
    / df_ventas_zona["ventas_mes_anterior"] * 100
)

# --- Construcción del gráfico narrativo ---
colores = ["#c0392b" if zona == "Norte" else "#bdc3c7" for zona in df_ventas_zona["zona"]]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(df_ventas_zona["zona"], df_ventas_zona["ventas_mes_actual"], color=colores)

ax.set_title(
    "Ventas por zona — Octubre 2026\nLa zona Norte cayó 47% frente al mes anterior",
    fontsize=13, fontweight="bold", loc="left",
)
ax.set_ylabel("Ventas (miles de $)")
ax.spines[["top", "right"]].set_visible(False)

fila_norte = df_ventas_zona.loc[df_ventas_zona["zona"] == "Norte"].iloc[0]
ax.annotate(
    f"Cierre temporal del local\ndel shopping (obras) — {fila_norte['variacion_pct']:.0f}%",
    xy=("Norte", fila_norte["ventas_mes_actual"]),
    xytext=(0.15, 0.65), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#c0392b"),
    fontsize=10, color="#c0392b", fontweight="bold",
)

plt.tight_layout()
plt.savefig("storytelling_reporte_estatico.png", dpi=150)
```

**Lo que hay que remarcar en clase:** el mensaje ("por qué cayó Norte") queda escrito en la imagen. Si alguien lee el PDF sin escuchar al presentador, igual entiende el mensaje — eso es lo que distingue a un reporte narrativo de un simple gráfico técnico.

---

### 1.2 Dashboard Interactivo

**Consigna:** el gerente de RRHH quiere explorar por su cuenta la rotación de personal por departamento, sin pedirle a un analista un gráfico nuevo cada vez que cambia la pregunta.

**Idea pedagógica clave:** acá NO elegimos una única historia. Le damos al usuario "libertad y control" (principio de usabilidad) mediante un filtro. Se simula con `Plotly` usando `updatemenus`, sin necesidad de levantar un servidor Dash — el HTML resultante ya es interactivo por sí solo.

```python
import pandas as pd
import plotly.graph_objects as go

# --- Dataset sandbox: rotación de personal mensual por departamento ---
departamentos = ["Ventas", "Sistemas", "Producción", "RRHH"]
meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
datos_rotacion = {
    "Ventas":     [4.2, 5.1, 6.8, 7.5, 6.0, 5.4],
    "Sistemas":   [1.1, 1.0, 1.4, 2.8, 3.1, 2.6],
    "Producción": [3.0, 3.2, 2.9, 3.1, 3.4, 3.0],
    "RRHH":       [0.8, 0.9, 0.7, 1.0, 0.6, 0.8],
}

fig = go.Figure()
for i, depto in enumerate(departamentos):
    fig.add_trace(go.Scatter(
        x=meses, y=datos_rotacion[depto],
        mode="lines+markers", name=depto, visible=(i == 0),
    ))

# El menú desplegable le da "control" al usuario: alterna qué traza
# está visible, sin tocar código ni recargar la página.
botones = []
for i, depto in enumerate(departamentos):
    visibilidad = [False] * len(departamentos)
    visibilidad[i] = True
    botones.append(dict(
        label=depto, method="update",
        args=[{"visible": visibilidad}, {"title": f"Rotación de personal — {depto}"}],
    ))

fig.update_layout(
    title="Rotación de personal — Ventas",
    updatemenus=[dict(active=0, buttons=botones, x=1.15, y=1.0, xanchor="left")],
    yaxis_title="Tasa de rotación (%)", xaxis_title="Mes (2026)",
    template="plotly_white",
)

fig.write_html("storytelling_dashboard_interactivo.html")
```

**Riesgo a discutir en clase:** si en vez de un filtro agregáramos 10 métricas simultáneas en pantalla, el "ruido visual" arruinaría la usabilidad. Por eso el ejemplo muestra **una sola métrica a la vez**.

---

### 1.3 Scrollytelling

**Consigna:** un medio digital quiere un artículo interactivo donde, a medida que el lector hace scroll, el gráfico de deforestación se va "dibujando" año a año, reforzando la conexión emocional con el dato.

**Idea pedagógica clave — por qué es "técnicamente costoso":** el scroll es un evento del navegador (JavaScript), no de Python. Python solo puede preparar los datos e inyectarlos en una plantilla HTML; quien realmente escucha el scroll y redibuja el gráfico es un poco de JavaScript (`IntersectionObserver`) sobre `Plotly.js`. Esta es una versión mínima con fines didácticos — en producción se usarían librerías especializadas como Scrollama.js.

```python
import json
import pandas as pd

# --- Dataset sandbox: superficie deforestada acumulada (miles de ha) ---
df_deforestacion = pd.DataFrame({
    "anio": list(range(2016, 2026)),
    "hectareas_miles": [120, 145, 168, 210, 250, 300, 355, 410, 470, 540],
})
anios = df_deforestacion["anio"].tolist()
valores = df_deforestacion["hectareas_miles"].tolist()

textos_narrativos = [
    "En 2016, la deforestación avanzaba a un ritmo relativamente controlado.",
    "Hacia 2020, el ritmo se acelera: cada año se pierde más superficie que el anterior.",
    "En 2025, la superficie acumulada casi cuadriplica a la de 2016.",
]

html_template = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8" />
<title>Scrollytelling — Deforestación (demo didáctica)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; }}
  #grafico-container {{ position: sticky; top: 0; height: 100vh; }}
  .seccion {{ height: 90vh; display: flex; align-items: center; padding: 0 8%; box-sizing: border-box; }}
  .seccion p {{ font-size: 1.5rem; max-width: 480px; background: rgba(255,255,255,0.92); padding: 1rem; }}
</style></head>
<body>
<div id="grafico-container"><div id="grafico" style="width:100%;height:100%;"></div></div>
<div style="margin-top:-100vh;">{secciones_html}</div>
<script>
const anios = {anios_json};
const valores = {valores_json};

function dibujarHasta(indice) {{
    Plotly.newPlot('grafico', [{{
        x: anios.slice(0, indice + 1), y: valores.slice(0, indice + 1),
        type: 'scatter', mode: 'lines+markers', fill: 'tozeroy', line: {{color: '#c0392b'}}
    }}], {{
        title: 'Superficie deforestada acumulada (miles de ha)',
        xaxis: {{range: [{anio_min}, {anio_max}]}}, yaxis: {{range: [0, {valor_max}]}}
    }});
}}
dibujarHasta(2);

const observador = new IntersectionObserver((entradas) => {{
    entradas.forEach((entrada) => {{
        if (entrada.isIntersecting) dibujarHasta(parseInt(entrada.target.dataset.indice, 10));
    }});
}}, {{ threshold: 0.6 }});
document.querySelectorAll('.seccion').forEach((el) => observador.observe(el));
</script></body></html>
"""

indices_por_escena = [2, 6, len(anios) - 1]
secciones_html = "\n".join(
    f'<div class="seccion" data-indice="{idx}"><p>{texto}</p></div>'
    for idx, texto in zip(indices_por_escena, textos_narrativos)
)

html_final = html_template.format(
    secciones_html=secciones_html, anios_json=json.dumps(anios), valores_json=json.dumps(valores),
    anio_min=min(anios), anio_max=max(anios), valor_max=max(valores) * 1.1,
)

with open("storytelling_scrollytelling_demo.html", "w", encoding="utf-8") as f:
    f.write(html_final)
```

Abrí el HTML resultante en el navegador y hacé scroll: vas a ver cómo el gráfico se va completando por secciones — esa es, en miniatura, la mecánica de un artículo interactivo real.

---

## Parte 2 — Análisis Espacial y Mapas GIS

Los cuatro ejercicios recorren un caso único de logística en Buenos Aires, de menor a mayor complejidad técnica.

### 2.1 Modelo Vectorial (Puntos) — Mapa Interactivo con Folium

**Consigna:** visualizar los 5 centros de distribución con marcadores explorables (zoom libre), en WGS 84 (EPSG:4326), el CRS geográfico estándar.

```python
import pandas as pd
import folium

data = {
    "ID_Centro": ["C-01", "C-02", "C-03", "C-04", "C-05"],
    "Nombre": ["Centro Norte", "Centro Sur", "Centro Oeste", "Centro Este", "Centro Central"],
    "Latitud": [-34.54, -34.65, -34.62, -34.60, -34.6037],
    "Longitud": [-58.46, -58.38, -58.50, -58.36, -58.3816],
    "Capacidad_Toneladas": [500, 350, 420, 280, 800],
}
df_centros = pd.DataFrame(data)

mapa_logistica = folium.Map(location=[-34.6037, -58.3816], zoom_start=11)
for _, fila in df_centros.iterrows():
    folium.Marker(
        location=[fila["Latitud"], fila["Longitud"]],
        popup=f"{fila['Nombre']} - Capacidad: {fila['Capacidad_Toneladas']}t",
        tooltip="Click para más info",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(mapa_logistica)

mapa_logistica.save("mapa_centros_distribucion.html")
```

---

### 2.2 CRS y Cálculo de Áreas con GeoPandas

**Consigna:** calcular el área real (km²) de dos zonas de entrega. El punto pedagógico central: los vértices están en grados (WGS 84), un CRS esférico que **no sirve para medir área directamente** — hay que reproyectar a un CRS plano (EPSG:3857) antes.

```python
import geopandas as gpd
from shapely.geometry import Polygon

zona_a_coords = [(-58.40, -34.60), (-58.38, -34.60), (-58.38, -34.62), (-58.40, -34.62)]
zona_b_coords = [(-58.45, -34.55), (-58.42, -34.55), (-58.42, -34.58), (-58.45, -34.58)]

poligono_a = Polygon(zona_a_coords)
poligono_b = Polygon(zona_b_coords)

gdf = gpd.GeoDataFrame(
    {"Nombre_Zona": ["Zona A", "Zona B"], "geometry": [poligono_a, poligono_b]},
    crs="EPSG:4326",
)

# Reproyección a un CRS proyectado (metros) para poder calcular área sin distorsión
gdf_proyectado = gdf.to_crs("EPSG:3857")
gdf_proyectado["Area_km2"] = gdf_proyectado.geometry.area / 10**6
print(gdf_proyectado[["Nombre_Zona", "Area_km2"]])
```

Resultado real obtenido al correrlo:

| Nombre_Zona | Area_km2 |
|---|---|
| Zona A | 6.02 |
| Zona B | 13.54 |

---

### 2.3 Storytelling + GIS — Mapa de Burbujas de Rendimiento Comercial

**Consigna:** mapa donde la ubicación indica la sucursal, el tamaño de la burbuja el volumen de ventas y el color el margen de ganancia.

> Nota: `px.scatter_map` es la función vigente en Plotly 6+/7 (reemplaza a `scatter_mapbox`, que queda deprecada y ya no requiere token de Mapbox para el estilo `carto-positron`). Si tu versión de Plotly es anterior, usá `px.scatter_mapbox` con `mapbox_style` en vez de `map_style`.

```python
import pandas as pd
import plotly.express as px

data_ventas = {
    "Sucursal": ["Sur_01", "Norte_01", "Oeste_01", "Este_01", "Centro_01"],
    "Latitud": [-34.65, -34.54, -34.62, -34.60, -34.6037],
    "Longitud": [-58.38, -58.46, -58.50, -58.36, -58.3816],
    "Volumen_Ventas": [150000, 420000, 210000, 95000, 600000],
    "Margen_Ganancia": [12.5, 24.0, 15.3, 8.2, 30.1],
}
df_ventas = pd.DataFrame(data_ventas)

fig = px.scatter_map(
    df_ventas, lat="Latitud", lon="Longitud",
    color="Margen_Ganancia", size="Volumen_Ventas", hover_name="Sucursal",
    hover_data={"Latitud": False, "Longitud": False, "Volumen_Ventas": True},
    color_continuous_scale=px.colors.sequential.Plasma, size_max=30, zoom=10.5,
    title="Rendimiento Comercial por Sucursal (Tamaño = Ventas, Color = Margen %)",
    map_style="carto-positron",
)
fig.write_html("mapa_burbujas_ventas.html")
```

---

### 2.4 Animación de un Trayecto Logístico

**Consigna:** representar la evolución temporal de un trayecto de entrega, punto a punto, en un mapa 2D — algo que un mapa estático no puede transmitir.

> Nota técnica: se guarda como GIF con `PillowWriter`, incluido en `matplotlib`. Es más portable que `writer='imagemagick'` porque no depende de instalar ImageMagick por fuera de Python.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

frames_totales = 50
inicio = np.array([-58.40, -34.60])
destino = np.array([-58.45, -34.55])

longitudes = np.linspace(inicio[0], destino[0], frames_totales)
latitudes = np.linspace(inicio[1], destino[1], frames_totales)
df_ruta = pd.DataFrame({"Longitud": longitudes, "Latitud": latitudes})

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-58.47, -58.38)
ax.set_ylim(-34.62, -34.53)
ax.set_title("Simulación de Trayecto de Reparto")
ax.grid(True, linestyle="--", alpha=0.6)
ax.plot(df_ruta["Longitud"], df_ruta["Latitud"], color="gray", linestyle="dashed", label="Ruta Planificada")
vehiculo, = ax.plot([], [], "ro", markersize=10, label="Posición Actual")
ax.legend()

def init():
    vehiculo.set_data([], [])
    return (vehiculo,)

def update(frame):
    x = df_ruta.loc[frame, "Longitud"]
    y = df_ruta.loc[frame, "Latitud"]
    vehiculo.set_data([x], [y])
    return (vehiculo,)

ani = FuncAnimation(fig, update, frames=frames_totales, init_func=init, interval=100, blit=True)
ani.save("gis_animacion_ruta.gif", writer=PillowWriter(fps=10))
```

---

## Parte 3 — Taxonomía del Análisis Estadístico (Univariado, Bivariado, Multivariado)

Acá el enfoque didáctico es distinto: en vez de tres consignas separadas, se usa **un único dataset de 150 propiedades inmobiliarias** (`dataset_propiedades.csv`) y se lo recorre con las tres lentes, para que quede clarísima la diferencia entre "mirar una variable", "mirar dos" y "mirar varias a la vez".

### 3.1 El dataset (150 filas, desde CSV)

El dataset ya está armado en `dataset_propiedades.csv` — no hace falta generarlo de nuevo. Si querés reconstruirlo desde cero (por ejemplo para cambiar la semilla o el tamaño de muestra), el script que lo generó es este:

```python
# taxonomia_0_generar_dataset.py — se corre UNA sola vez
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_PROPIEDADES = 150
rng = np.random.default_rng(RANDOM_SEED)

barrios = rng.choice(
    ["Palermo", "Belgrano", "Caballito", "Flores", "Recoleta"],
    size=N_PROPIEDADES, p=[0.25, 0.2, 0.2, 0.2, 0.15],
)
metros_cuadrados = rng.normal(loc=75, scale=25, size=N_PROPIEDADES).clip(28, 220)
ambientes = np.round(metros_cuadrados / 28 + rng.normal(0, 0.4, N_PROPIEDADES)).clip(1, 7)
antiguedad_anios = rng.integers(0, 60, N_PROPIEDADES)
distancia_centro_km = rng.exponential(scale=4.0, size=N_PROPIEDADES).clip(0.2, 22)

# El precio depende linealmente de m2 y ambientes, e inversamente de la
# distancia al centro y la antigüedad, más ruido — para que las
# correlaciones que aparecen más adelante tengan una causa real.
precio_usd = (
    metros_cuadrados * 1450 + ambientes * 3200
    - antiguedad_anios * 380 - distancia_centro_km * 2100
    + rng.normal(0, 9000, N_PROPIEDADES)
).clip(18000, None)

df_propiedades = pd.DataFrame({
    "barrio": barrios,
    "metros_cuadrados": metros_cuadrados.round(1),
    "ambientes": ambientes.astype(int),
    "antiguedad_anios": antiguedad_anios,
    "distancia_centro_km": distancia_centro_km.round(2),
    "precio_usd": precio_usd.round(0),
})
df_propiedades.to_csv("dataset_propiedades.csv", index=False)
```

**Desde acá en adelante, todos los ejemplos parten del CSV** — así se comporta cualquier análisis real, donde el dataset llega como archivo entregado por otra área y no se regenera cada vez:

```python
import pandas as pd

df_propiedades = pd.read_csv("dataset_propiedades.csv")
print(df_propiedades.head())
```

### 3.2 Univariado ($d=1$) — `precio_usd` en aislamiento

```python
media_precio = df_propiedades["precio_usd"].mean()
mediana_precio = df_propiedades["precio_usd"].median()

# OJO: pandas .var()/.std() usan ddof=1 por defecto -> dividen por (n-1),
# no por n. Es la corrección de Bessel: al usar la MEDIA MUESTRAL (que ya
# "consumió" un grado de libertad) para calcular la varianza, dividir por
# n subestimaría sistemáticamente la varianza poblacional real.
varianza_precio = df_propiedades["precio_usd"].var(ddof=1)
desvio_precio = df_propiedades["precio_usd"].std(ddof=1)
```

Resultado real obtenido:
- Media: **96.488 USD** — Mediana: **94.777 USD**
- Desvío estándar: **40.189 USD** (dividiendo por n-1)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(df_propiedades["precio_usd"], bins=20, color="#2980b9", edgecolor="white")
axes[0].axvline(media_precio, color="#c0392b", linestyle="--", label=f"Media: {media_precio:,.0f}")
axes[0].axvline(mediana_precio, color="#27ae60", linestyle="--", label=f"Mediana: {mediana_precio:,.0f}")
axes[0].set_title("Distribución del precio (histograma)")
axes[0].legend(fontsize=8)

axes[1].boxplot(df_propiedades["precio_usd"], vert=True)
axes[1].set_title("Precio (boxplot) — detección de outliers")
plt.tight_layout()
plt.savefig("taxonomia_1_univariado.png", dpi=150)
```

**Limitación a remarcar:** con esto solo sabemos "cómo se comporta el precio en soledad" — nada nos dice todavía si m² o la antigüedad influyen.

---

### 3.3 Bivariado ($d=2$) — `metros_cuadrados` vs `precio_usd`

```python
covarianza_m2_precio = df_propiedades[["metros_cuadrados", "precio_usd"]].cov().iloc[0, 1]
correlacion_m2_precio = df_propiedades["metros_cuadrados"].corr(df_propiedades["precio_usd"])
```

Resultado real: **covarianza ≈ 926.417**, **correlación de Pearson ≈ 0.94** (relación lineal fuerte, como se esperaba por cómo se construyó el dataset).

**La trampa de la covarianza y la independencia**, con una relación puramente parabólica armada a propósito (esta parte no viene del CSV — es un experimento aparte, generado en el momento, solo para visualizar la trampa conceptual):

```python
import numpy as np

rng = np.random.default_rng(42)
x_parabola = rng.uniform(-10, 10, 200)
y_parabola = x_parabola**2 + rng.normal(0, 2, 200)
covarianza_parabola = np.cov(x_parabola, y_parabola, ddof=1)[0, 1]
```

Resultado real: **covarianza ≈ -5** (prácticamente cero comparada con la escala de los datos), a pesar de que `y` depende **totalmente** de `x` — solo que la relación es cuadrática, no lineal. Esto es exactamente el punto de la "Alerta Crítica" del apunte original: covarianza cero (o cercana) no implica independencia, solo ausencia de relación lineal.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(df_propiedades["metros_cuadrados"], df_propiedades["precio_usd"], alpha=0.6, color="#2980b9")
axes[0].set_title(f"m² vs precio (relación lineal real, r={correlacion_m2_precio:.2f})")

axes[1].scatter(x_parabola, y_parabola, alpha=0.6, color="#8e44ad")
axes[1].set_title(f"Trampa: relación parabólica (cov≈{covarianza_parabola:.1f}, no lineal)")
plt.tight_layout()
plt.savefig("taxonomia_2_bivariado.png", dpi=150)
```

---

### 3.4 Multivariado ($d \ge 3$) — Heatmap y PCA

**Mapa de calor de correlaciones** entre las 5 variables numéricas:

```python
import seaborn as sns

columnas_numericas = ["metros_cuadrados", "ambientes", "antiguedad_anios", "distancia_centro_km", "precio_usd"]
matriz_correlacion = df_propiedades[columnas_numericas].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Mapa de calor — correlación entre variables de la propiedad")
plt.tight_layout()
plt.savefig("taxonomia_3_heatmap.png", dpi=150)
```

**El peligro de las escalas en PCA**, comparando sin estandarizar vs. estandarizando:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df_propiedades[["metros_cuadrados", "ambientes", "antiguedad_anios", "distancia_centro_km"]].values

pca_sin_escalar = PCA(n_components=2, random_state=RANDOM_SEED).fit(X)

X_escalado = StandardScaler().fit_transform(X)
pca_escalado = PCA(n_components=2, random_state=RANDOM_SEED).fit(X_escalado)
```

Resultado real obtenido (cargas del primer componente principal, PC1):

| Variable | Sin estandarizar | Estandarizado |
|---|---|---|
| metros_cuadrados | **0.999** | 0.702 |
| ambientes | 0.032 | 0.700 |
| antiguedad_anios | -0.038 | -0.001 |
| distancia_centro_km | 0.016 | 0.131 |

**Esto es exactamente la trampa que hay que discutir en clase:** sin estandarizar, `metros_cuadrados` (que se mueve en decenas) captura prácticamente el 100% del PC1 — como si fuera la única variable relevante, algo falso. Al estandarizar (media 0, desvío 1 en todas las variables), el peso se reparte de forma mucho más realista entre `metros_cuadrados` y `ambientes`, las dos variables que efectivamente están correlacionadas entre sí.

---

## Cierre para el aula — Scrollytelling con el mismo dataset

Cerramos integrando las tres partes: usamos el `dataset_propiedades.csv` de la Parte 3 (no uno nuevo) para armar un scrollytelling que va revelando, barrio por barrio, el precio promedio de las propiedades — igual que se hizo con la deforestación en 1.3, pero ahora con datos que los alumnos ya trabajaron.

```python
import json
import pandas as pd

df_propiedades = pd.read_csv("dataset_propiedades.csv")

resumen_barrios = (
    df_propiedades.groupby("barrio")
    .agg(
        precio_promedio=("precio_usd", "mean"),
        m2_promedio=("metros_cuadrados", "mean"),
        cantidad=("precio_usd", "size"),
    )
    .round(0)
    .sort_values("precio_promedio", ascending=False)
    .reset_index()
)

barrios_ordenados = resumen_barrios["barrio"].tolist()
precios = resumen_barrios["precio_promedio"].tolist()

# Un párrafo narrativo por barrio, generado a partir de los propios datos
# (no hardcodeado): si cambia el CSV, el texto se actualiza solo.
textos_narrativos = []
for i, fila in resumen_barrios.iterrows():
    if i == 0:
        texto = (
            f"<strong>{fila['barrio']}</strong> encabeza el ranking, con un precio "
            f"promedio de <strong>${fila['precio_promedio']:,.0f}</strong> sobre "
            f"{int(fila['cantidad'])} propiedades relevadas."
        )
    else:
        diferencia_pct = (fila["precio_promedio"] / precios[0] - 1) * 100
        texto = (
            f"<strong>{fila['barrio']}</strong> promedia "
            f"<strong>${fila['precio_promedio']:,.0f}</strong> "
            f"({diferencia_pct:.0f}% respecto al líder), con propiedades de "
            f"{fila['m2_promedio']:.0f} m² en promedio."
        )
    textos_narrativos.append(texto)
```

El HTML se arma con la misma técnica que en 1.3: un `<div>` con el gráfico en `position: sticky`, una sección por barrio, y un `IntersectionObserver` en JavaScript que redibuja las barras (agregando una más, en rojo) cada vez que una sección nueva entra en pantalla.

```python
html_template = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8" />
<title>Scrollytelling — Precio promedio por barrio</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; }}
  #grafico-container {{ position: sticky; top: 0; height: 100vh; }}
  .seccion {{ height: 90vh; display: flex; align-items: center; padding: 0 8%; box-sizing: border-box; }}
  .seccion p {{ font-size: 1.4rem; max-width: 480px; line-height: 1.4; background: rgba(255,255,255,0.92); padding: 1rem; }}
</style></head>
<body>
<div id="grafico-container"><div id="grafico" style="width:100%;height:100%;"></div></div>
<div style="margin-top:-100vh;">{secciones_html}</div>
<script>
const barrios = {barrios_json};
const precios = {precios_json};

function dibujarHasta(indice) {{
    const colores = barrios.map((_, i) => i <= indice ? '#c0392b' : '#bdc3c7');
    Plotly.newPlot('grafico', [{{
        x: barrios.slice(0, indice + 1), y: precios.slice(0, indice + 1),
        type: 'bar', marker: {{color: colores.slice(0, indice + 1)}}
    }}], {{
        title: 'Precio promedio por barrio (USD)',
        xaxis: {{range: [-0.5, barrios.length - 0.5]}},
        yaxis: {{range: [0, Math.max(...precios) * 1.15]}}
    }});
}}
dibujarHasta(0);

const observador = new IntersectionObserver((entradas) => {{
    entradas.forEach((entrada) => {{
        if (entrada.isIntersecting) dibujarHasta(parseInt(entrada.target.dataset.indice, 10));
    }});
}}, {{ threshold: 0.6 }});
document.querySelectorAll('.seccion').forEach((el) => observador.observe(el));
</script></body></html>
"""

secciones_html = "\n".join(
    f'<div class="seccion" data-indice="{i}"><p>{texto}</p></div>'
    for i, texto in enumerate(textos_narrativos)
)

html_final = html_template.format(
    secciones_html=secciones_html,
    barrios_json=json.dumps(barrios_ordenados),
    precios_json=json.dumps(precios),
)

with open("scrollytelling_precio_por_barrio.html", "w", encoding="utf-8") as f:
    f.write(html_final)
```

Resultado real al correrlo sobre el dataset entregado:

| Barrio | Precio promedio | m² promedio | Cantidad |
|---|---|---|---|
| Recoleta | $106.781 | 77 | 15 |
| Flores | $103.512 | 79 | 34 |
| Caballito | $97.969 | 77 | 33 |
| Belgrano | $90.006 | 73 | 31 |
| Palermo | $89.970 | 70 | 37 |

**Punto para discutir en clase, cerrando el círculo con la Parte 3:** en este dataset sintético, el precio **no depende del barrio** — el modelo de generación solo usa m², ambientes, antigüedad y distancia al centro. Las diferencias que aparecen acá son producto de qué mezcla de propiedades tocó en la muestra de cada barrio, no de una "prima" real por ubicación. Es la misma pregunta que plantea la trampa de la covarianza: *¿esta diferencia visual es una relación real, o es ruido de muestreo disfrazado de historia?* Un buen cierre es pedirles a los alumnos que, antes de creerle al gráfico, corran el análisis multivariado de la sección 3.4 agrupando por barrio y vean si la diferencia se sostiene al controlar por m² y distancia.