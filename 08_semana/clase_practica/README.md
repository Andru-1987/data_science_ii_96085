# Análisis Univariado y Bivariado — Caso Práctico: Mercado Inmobiliario

Armé un dataset sintético de **600 propiedades** diseñado a propósito para que cada fenómeno teórico del documento aparezca de forma clara y verificable. No es un dataset real de Kaggle, pero está construido con relaciones matemáticas conocidas (lognormales, efectos cuadráticos, ruido controlado), así que los resultados son 100% reproducibles con el código que te dejo.

## El dataset: `propiedades.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `barrio` | Categórica | Norte, Centro, Sur, Costa |
| `tipo_propiedad` | Categórica | Casa, Departamento, PH |
| `metros_cuadrados` | Numérica continua | Generada lognormal (asimétrica, como en la vida real) |
| `ambientes` | Numérica discreta | Derivada de m² |
| `antiguedad_anios` | Numérica | 0 a 60 años |
| `precio_usd` | Numérica continua | Variable objetivo, con outliers de lujo inyectados a propósito |
| `impuestos_anuales_usd` | Numérica | Construida correlacionada fuertemente con `precio_usd` |
| `dias_en_mercado` | Numérica | Tiempo de venta |

```python
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
```

---

## 8.1 Análisis Univariado

### Variable numérica: `precio_usd`

```python
df["precio_usd"].describe()
```

| Métrica | Valor |
|---|---|
| Media | **USD 90.347** |
| Mediana | **USD 80.379** |
| Desv. estándar | USD 51.950 |
| Q1 / Q3 | 55.705 / 110.013 |
| IQR | 54.308 |
| Límite superior (Q3 + 1.5·IQR) | 191.475 |
| Outliers detectados | **23 (3.8%)** |

**Preguntas guía → Hallazgos:**

- *¿La media y la mediana coinciden?* No — la media (90.347) es **~12% mayor** que la mediana (80.379). Esto ya nos avisa, sin graficar nada, que la distribución no es simétrica.
- *¿Hay outliers?* Sí, 23 propiedades (3.8%) superan el límite de Q3+1.5·IQR. Corresponden en su mayoría a las propiedades "de lujo" que inyectamos deliberadamente.
- *¿Qué forma tiene la distribución?* Un histograma mostraría una cola larga a la derecha (precios muy altos poco frecuentes) — coherente con el hallazgo de skewness que calculamos en 8.3.

### Variable categórica: `tipo_propiedad`

| tipo_propiedad | Frecuencia | % |
|---|---|---|
| Departamento | 258 | 43% |
| Casa | 216 | 36% |
| PH | 126 | 21% |

**Hallazgo:** el mercado está dominado por departamentos; los PH son el segmento minoritario — dato relevante antes de comparar precios por tipo (8.2-B), porque ese grupo tendrá menos poder estadístico.

---

## 8.2 Análisis Bivariado

### A) Numérica vs Numérica — `metros_cuadrados` vs `precio_usd`

```python
from scipy import stats
r, p = stats.pearsonr(df["metros_cuadrados"], df["precio_usd"])
```

**Resultado:** r = **0.644**, p < 0.001

**Pregunta → Hallazgo:** *¿El tamaño explica el precio?* Sí, hay una correlación lineal fuerte y significativa: a mayor superficie, mayor precio. Pero r=0.64 (no 0.95) también nos dice que **el tamaño no es la única variable relevante** — barrio y tipo de propiedad también pesan, como veremos en (B).

### B) Categórica vs Numérica — `tipo_propiedad` vs `precio_usd` (ANOVA)

```python
grupos = [g["precio_usd"].values for _, g in df.groupby("tipo_propiedad")]
f, p = stats.f_oneway(*grupos)
```

| tipo_propiedad | Media | Mediana | Std |
|---|---|---|---|
| Casa | 99.205 | 86.187 | 57.081 |
| Departamento | 86.342 | 80.216 | 44.979 |
| PH | 83.361 | 69.492 | 54.321 |

**Resultado ANOVA:** F = 5.12, p = 0.006

**Pregunta → Hallazgo:** *¿El tipo de propiedad influye en el precio?* Sí, la diferencia entre grupos es estadísticamente significativa (p<0.05): las casas son en promedio ~19% más caras que los PH. Un boxplot agrupado confirmaría visualmente que la distribución de "Casa" está desplazada hacia arriba.

### C) Categórica vs Categórica — `barrio` vs `tipo_propiedad` (Chi-cuadrado)

| barrio | Casa | Departamento | PH |
|---|---|---|---|
| Centro | 71 | 83 | 50 |
| Costa | 30 | 33 | 17 |
| Norte | 56 | 74 | 27 |
| Sur | 59 | 68 | 32 |

**Resultado:** Chi² = 3.46, p = 0.749

**Pregunta → Hallazgo:** *¿El tipo de propiedad se distribuye distinto según el barrio?* No — p=0.749 (muy por encima de 0.05) nos dice que **no hay asociación significativa**: la mezcla de tipos de propiedad es prácticamente la misma en los cuatro barrios. Esto es un hallazgo tan válido como encontrar dependencia: nos permite tratar `barrio` y `tipo_propiedad` como fuentes de variación independientes al modelar precio.

---

## 8.3 Profundización en Métricas Estadísticas

### Desviación estándar por barrio (volatilidad de precios)

| barrio | Media | Std |
|---|---|---|
| Costa | 138.985 | **55.229** |
| Norte | 105.487 | 52.767 |
| Centro | 84.573 | 44.541 |
| Sur | 58.333 | **30.136** |

**Hallazgo:** Costa no solo tiene los precios más altos, también la mayor dispersión — es el segmento de "mayor riesgo/volatilidad" del mercado, análogo al caso financiero del documento. Sur es el más predecible.

### Skewness y Kurtosis de `precio_usd`

```python
stats.skew(df["precio_usd"])      # 2.05
stats.kurtosis(df["precio_usd"])  # 6.65 (exceso, Fisher)
```

**Hallazgo:** Skewness = **2.05** (>1, asimetría positiva fuerte) y curtosis en exceso = **6.65** (>0, leptocúrtica, colas pesadas). Traducido a decisiones de modelado: si vas a entrenar una regresión lineal sobre `precio_usd`, **conviene aplicar `log(precio_usd)`** antes de entrenar, tal como indica la teoría.

### Pearson vs Spearman vs Kendall — `antiguedad_anios` vs `precio_usd`

```python
stats.pearsonr(df["antiguedad_anios"], df["precio_usd"])   # r=0.129
stats.spearmanr(df["antiguedad_anios"], df["precio_usd"])  # rho=0.147
stats.kendalltau(df["antiguedad_anios"], df["precio_usd"]) # tau=0.099
```

**Hallazgo clave (y matiz importante que no está explícito en la teoría):** las tres correlaciones son **bajas** (0.10-0.15). Si te quedaras solo con estos números, concluirías que la antigüedad casi no influye en el precio. Pero mirá el promedio por tramos de edad:

| Tramo antigüedad | Precio medio |
|---|---|
| 0-10 años | 86.527 |
| 10-20 | 86.963 |
| 20-30 | **80.522** (mínimo) |
| 30-40 | 84.634 |
| 40-50 | 94.198 |
| 50-60 | **113.709** (máximo) |

Hay una **forma de U clarísima** (las propiedades muy nuevas o muy antiguas/reformadas valen más que las de mediana edad), pero ni Pearson ni Spearman ni Kendall la detectan bien, porque **ninguna de las tres captura relaciones no monótonas** (Spearman y Kendall solo mejoran sobre Pearson cuando la relación es monótona, como un crecimiento exponencial — no cuando sube y luego baja, o viceversa). Este es exactamente el caso del Tip 1 del documento, y muestra por qué "graficar antes de descartar" es una regla no negociable.

### Multicolinealidad — `impuestos_anuales_usd` vs `precio_usd`

```python
df[["metros_cuadrados","precio_usd","impuestos_anuales_usd"]].corr()
```

| | metros_cuadrados | precio_usd | impuestos_anuales_usd |
|---|---|---|---|
| metros_cuadrados | 1.00 | 0.64 | 0.65 |
| precio_usd | 0.64 | 1.00 | **0.98** |
| impuestos_anuales_usd | 0.65 | 0.98 | 1.00 |

**Hallazgo:** r=0.98 entre precio e impuestos — muy por encima del umbral de 0.85 del documento. Este es un caso de manual de multicolinealidad: si vas a predecir `precio_usd` con un modelo de regresión, **`impuestos_anuales_usd` es redundante y debería eliminarse** (o usarse solo una de las dos), porque prácticamente no aporta información nueva sobre el precio que no esté ya en la variable objetivo misma.

---

## 8.4 Tips en acción

### Tip 1 — Ya cubierto arriba (antigüedad vs precio, relación en U)

### Tip 2 — Paradoja de Simpson

Con el mismo espíritu del dataset (dos "agencias" vendiendo propiedades, y la variable "se vendió en menos de 60 días"), armé una tabla clásica de paradoja de Simpson:

| | Departamento | | Casa | | **Total** |
|---|---|---|---|---|---|
| | Vendidas rápido / Total | % | Vendidas rápido / Total | % | % |
| **Agencia X** | 81/87 | **93.1%** | 192/263 | **73.0%** | 78.0% |
| **Agencia Y** | 234/270 | 86.7% | 55/80 | 68.8% | **82.6%** |

**Pregunta → Hallazgo:** *¿Qué agencia vende más rápido?* Mirando el total, Agencia Y parece mejor (82.6% vs 78.0%). Pero **segmentando por tipo de propiedad, Agencia X es mejor en ambos segmentos** (93.1%>86.7% en Departamento, 73.0%>68.8% en Casa). La paradoja ocurre porque Agencia Y vendió mucho más volumen de Departamentos (el segmento "fácil" de vender rápido), inflando su promedio total. **Conclusión práctica:** nunca decidas qué agencia contratar mirando solo el total agregado — siempre desagregá por la variable oculta (aquí, el mix de tipo de propiedad que maneja cada una).

### Tip 3 — Ya cubierto arriba (multicolinealidad impuestos/precio)

---

## Resumen de hallazgos para cerrar la clase

1. `precio_usd` es asimétrico y leptocúrtico → requiere transformación log antes de modelar.
2. El tamaño (m²) explica el precio moderadamente (r=0.64); el tipo de propiedad también influye (ANOVA significativo); barrio y tipo son independientes entre sí.
3. Costa es el barrio más caro y más volátil; Sur el más económico y estable.
4. La antigüedad **no tiene correlación lineal ni monótona** con el precio, pero sí una relación en U real — lección sobre las limitaciones de los tres coeficientes de correlación.
5. Impuestos y precio están multicolineales (r=0.98) — candidato a eliminar en un modelo predictivo.
6. La paradoja de Simpson recuerda que un ranking agregado puede mentir si no se desagrega por la variable de confusión.


--- 
### Desarrollo de Charts

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Asegurarse de tener el dataset original en un DataFrame
df = pd.DataFrame({
    "barrio": barrio,
    "tipo_propiedad": tipo,
    "metros_cuadrados": m2,
    "antiguedad_anios": antiguedad,
    "precio_usd": precio_usd,
    "impuestos_anuales_usd": impuestos_anuales_usd
})

# Tema por defecto para gráficos limpios
template = "plotly_white"

# ==========================================
# 8.1 ANÁLISIS UNIVARIADO
# ==========================================

# 1. Distribución de precio_usd (Histograma + Boxplot)
fig_precio = px.histogram(
    df, x="precio_usd", marginal="box", nbins=50,
    title="<b>1. Distribución de Precios: Asimetría y Outliers de Lujo</b><br><sup>La media (línea roja) es arrastrada hacia la derecha por los outliers, separándose de la mediana (línea verde).</sup>",
    labels={"precio_usd": "Precio (USD)"},
    color_discrete_sequence=["#1f77b4"],
    template=template
)
fig_precio.add_vline(x=df["precio_usd"].mean(), line_dash="dash", line_color="red", annotation_text="Media")
fig_precio.add_vline(x=df["precio_usd"].median(), line_dash="dash", line_color="green", annotation_text="Mediana")
fig_precio.show()

# 2. Frecuencia de tipo_propiedad (Gráfico de Barras)
conteo_tipo = df['tipo_propiedad'].value_counts().reset_index()
conteo_tipo.columns = ['Tipo de Propiedad', 'Cantidad']
fig_tipo = px.bar(
    conteo_tipo, x='Tipo de Propiedad', y='Cantidad', text='Cantidad',
    title="<b>2. Composición del Mercado Inmobiliario</b><br><sup>Dominan los Departamentos; el segmento de PH es minoritario.</sup>",
    color='Tipo de Propiedad',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    template=template
)
fig_tipo.update_traces(textposition='outside')
fig_tipo.show()

# ==========================================
# 8.2 ANÁLISIS BIVARIADO
# ==========================================

# 3. Numérica vs Numérica: m2 vs precio (Scatter con tendencia)
fig_m2_precio = px.scatter(
    df, x="metros_cuadrados", y="precio_usd", color="tipo_propiedad",
    trendline="ols", opacity=0.7,
    title="<b>3. Tamaño vs Precio (r=0.644)</b><br><sup>Relación lineal clara, pero la dispersión confirma que el m² no es el único factor determinante.</sup>",
    labels={"metros_cuadrados": "Metros Cuadrados", "precio_usd": "Precio (USD)"},
    template=template
)
fig_m2_precio.show()

# 4. Categórica vs Numérica: Tipo vs Precio (Boxplot / ANOVA)
fig_anova = px.box(
    df, x="tipo_propiedad", y="precio_usd", color="tipo_propiedad",
    title="<b>4. Precios según Tipo de Propiedad</b><br><sup>Diferencia significativa (ANOVA p=0.006): Las casas tienen una mediana superior y mayor alcance.</sup>",
    labels={"tipo_propiedad": "Tipo", "precio_usd": "Precio (USD)"},
    template=template
)
fig_anova.show()

# 5. Categórica vs Categórica: Barrio vs Tipo (Barras apiladas 100%)
fig_chi2 = px.histogram(
    df, x="barrio", color="tipo_propiedad", barnorm="percent", text_auto='.0f',
    title="<b>5. Mix Inmobiliario por Barrio (Chi² no significativo)</b><br><sup>La proporción de tipos de propiedad es prácticamente idéntica sin importar el barrio.</sup>",
    labels={"barrio": "Barrio", "percent": "Porcentaje (%)"},
    template=template
)
fig_chi2.show()

# ==========================================
# 8.3 PROFUNDIZACIÓN
# ==========================================

# 6. Volatilidad de precios por barrio (Violin plot)
fig_volatilidad = px.violin(
    df, x="barrio", y="precio_usd", box=True, color="barrio",
    title="<b>6. Volatilidad y Riesgo por Barrio</b><br><sup>La 'Costa' es el segmento más caro pero también el de mayor dispersión (riesgo).</sup>",
    labels={"barrio": "Barrio", "precio_usd": "Precio (USD)"},
    category_orders={"barrio": ["Costa", "Norte", "Centro", "Sur"]},
    template=template
)
fig_volatilidad.show()

# 7. Relación no lineal: Antigüedad vs Precio
# Agregamos línea LOWESS para capturar la forma en U
fig_antiguedad = px.scatter(
    df, x="antiguedad_anios", y="precio_usd", trendline="lowess",
    trendline_color_override="red", opacity=0.4,
    title="<b>7. El Punto Ciego de Pearson: Antigüedad vs Precio</b><br><sup>La línea de tendencia LOWESS revela la relación en forma de 'U' que las métricas lineales (r~0.12) ignoran.</sup>",
    labels={"antiguedad_anios": "Antigüedad (Años)", "precio_usd": "Precio (USD)"},
    template=template
)
fig_antiguedad.show()

# 8. Multicolinealidad (Mapa de calor de correlación)
matriz_corr = df[["metros_cuadrados", "precio_usd", "impuestos_anuales_usd"]].corr().round(2)
fig_corr = px.imshow(
    matriz_corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    title="<b>8. Multicolinealidad Detectada</b><br><sup>Correlación de 0.98 entre precio e impuestos. Se debe eliminar una en modelado predictivo.</sup>",
    template=template
)
fig_corr.show()

# ==========================================
# 8.4 TIP: PARADOJA DE SIMPSON
# ==========================================

# Datos recreados de la tabla teórica para visualizar la paradoja
df_simpson = pd.DataFrame({
    "Agencia": ["Agencia X", "Agencia X", "Agencia Y", "Agencia Y"],
    "Segmento": ["Departamento", "Casa", "Departamento", "Casa"],
    "Tasa_Venta_Rapida": [93.1, 73.0, 86.7, 68.8]
})

fig_simpson = px.bar(
    df_simpson, x="Segmento", y="Tasa_Venta_Rapida", color="Agencia", barmode="group",
    text="Tasa_Venta_Rapida",
    title="<b>9. Paradoja de Simpson en Acción</b><br><sup>Aunque Agencia Y tiene mejor promedio global (82.6% vs 78%), Agencia X domina en CADA segmento individual.</sup>",
    labels={"Tasa_Venta_Rapida": "% de Ventas Rápidas (<60 días)"},
    color_discrete_sequence=["#2ca02c", "#d62728"],
    template=template
)
fig_simpson.update_traces(texttemplate='%{text}%', textposition='outside')
# Líneas horizontales ilustrativas para los promedios globales
fig_simpson.add_hline(y=82.6, line_dash="dot", line_color="#d62728", annotation_text="Promedio Global Y (Falso Ganador)")
fig_simpson.add_hline(y=78.0, line_dash="dot", line_color="#2ca02c", annotation_text="Promedio Global X")
fig_simpson.show()

```
