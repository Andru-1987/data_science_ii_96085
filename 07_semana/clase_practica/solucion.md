### Resolucion del desafio

**Paso previo:** Carga de datos y librerías.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Cargamos el dataset generado por el script final
df_entregas = pd.read_csv("dataset_logistica_caba.csv")

```

**Solución 1: Storytelling (Reporte Estático)**

```python
# Preparación de datos agregados
df_volumen = df_entregas.groupby("zona").size().reset_index(name="entregas")
df_volumen = df_volumen.sort_values("entregas", ascending=False)

colores = ["#c0392b" if z == "Sur" else "#bdc3c7" for z in df_volumen["zona"]]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(df_volumen["zona"], df_volumen["entregas"], color=colores)

ax.set_title(
    "Volumen de Entregas por Zona (CABA y GBA)\nFuerte impacto operativo en la Zona Sur",
    fontsize=13, fontweight="bold", loc="left",
)
ax.set_ylabel("Cantidad de entregas")
ax.spines[["top", "right"]].set_visible(False)

entregas_sur = df_volumen.loc[df_volumen["zona"] == "Sur", "entregas"].values[0]
ax.annotate(
    "Caída del 35% frente al target\npor cortes recurrentes en Puente Pueyrredón",
    xy=("Sur", entregas_sur),
    xytext=(0.35, 0.65), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
    fontsize=10, color="#c0392b", fontweight="bold",
)

plt.tight_layout()
plt.savefig("solucion_1_estatico_caba.png", dpi=150)

```

**Solución 2: Storytelling (Dashboard Interactivo)**

```python
df_mensual = df_entregas.groupby(["mes", "zona"])["tiempo_entrega_min"].mean().reset_index()
meses_orden = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
zonas = df_mensual["zona"].unique()

fig = go.Figure()
for i, zona in enumerate(zonas):
    df_filtrado = df_mensual[df_mensual["zona"] == zona]
    fig.add_trace(go.Scatter(
        x=meses_orden, y=df_filtrado["tiempo_entrega_min"],
        mode="lines+markers", name=zona, visible=(i == 0)
    ))

botones = []
for i, zona in enumerate(zonas):
    visibilidad = [False] * len(zonas)
    visibilidad[i] = True
    botones.append(dict(
        label=zona, method="update",
        args=[{"visible": visibilidad}, {"title": f"Tiempo de Entrega Promedio — Zona {zona}"}]
    ))

fig.update_layout(
    title=f"Tiempo de Entrega Promedio — Zona {zonas[0]}",
    updatemenus=[dict(active=0, buttons=botones, x=1.15, y=1.0, xanchor="left")],
    yaxis_title="Minutos promedio", xaxis_title="Mes (2026)",
    template="plotly_white",
)
fig.write_html("solucion_2_interactivo_caba.html")

```

**Solución 3: Análisis Espacial (Mapa GIS sobre CABA)**

```python
# Tomamos una muestra para no saturar el navegador con miles de puntos
df_muestra = df_entregas.sample(350, random_state=42)

# Usamos px.scatter_map (Plotly 6+) centrado en el Obelisco
fig = px.scatter_map(
    df_muestra, lat="latitud", lon="longitud",
    color="satisfaccion_cliente", size="costo_envio_ars", hover_name="id_pedido",
    hover_data={"latitud": False, "longitud": False, "distancia_km": True},
    color_continuous_scale=px.colors.diverging.RdYlGn, size_max=15, 
    zoom=11, center={"lat": -34.6037, "lon": -58.3816}, # Centro exacto de CABA
    title="Mapa de Repartos Porteños (Tamaño = Costo ARS, Color = Satisfacción)",
    map_style="carto-positron",
)
fig.write_html("solucion_3_mapa_caba.html")

```

**Solución 4: Análisis Estadístico (Univariado en Kilómetros)**

```python
media_km = df_entregas["distancia_km"].mean()
mediana_km = df_entregas["distancia_km"].median()
desvio_km = df_entregas["distancia_km"].std(ddof=1)

print(f"Distancia Media: {media_km:.2f} km | Desvío: {desvio_km:.2f} km")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_entregas["distancia_km"], bins=25, color="#8e44ad", edgecolor="white")
ax.axvline(media_km, color="#c0392b", linestyle="--", label=f"Media: {media_km:.2f} km")
ax.axvline(mediana_km, color="#27ae60", linestyle="--", label=f"Mediana: {mediana_km:.2f} km")
ax.set_title("Distribución de la Distancia de los Recorridos (km)")
ax.set_xlabel("Kilómetros")
ax.set_ylabel("Frecuencia")
ax.legend()
plt.tight_layout()
plt.savefig("solucion_4_univariado_caba.png", dpi=150)

```

**Solución 5: Análisis Estadístico (Bivariado y Multivariado)**

```python
cols_numericas = ["distancia_km", "tiempo_entrega_min", "costo_envio_ars", "satisfaccion_cliente"]
matriz_corr = df_entregas[cols_numericas].corr()

fig, ax = plt.subplots(figsize=(8, 6))
# Usamos un mapa de calor para ver cómo los km y el tiempo arruinan la satisfacción
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlación: Impacto de los km y el tiempo en el negocio")
plt.tight_layout()
plt.savefig("solucion_5_heatmap_caba.png", dpi=150)

```