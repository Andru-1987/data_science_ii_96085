## 0. Dataset de ejemplo

Se generan 3 perfiles de clientes con variables correlacionadas entre sí (para que PCA tenga sentido) y 12 clientes atípicos (para que se note el efecto de los outliers).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_blobs

RNG = 42


def generar_clientes(seed=RNG):
    rng = np.random.default_rng(seed)
    cols = ["edad", "ingreso_anual", "gasto_promedio",
            "frecuencia_compras", "visitas_web_mes", "pct_descuento"]

    # (n, medias, desvíos) por perfil
    perfiles = {
        "Cazadores de ofertas": (240, [27, 38000, 60, 20, 30, 0.80],
                                      [4, 6000, 12, 4, 6, 0.07]),
        "Compradores premium":  (160, [45, 110000, 260, 8, 12, 0.10],
                                      [6, 15000, 40, 2, 3, 0.05]),
        "Ocasionales":          (200, [38, 65000, 110, 3, 5, 0.35],
                                      [8, 10000, 25, 1, 2, 0.10]),
    }

    partes = []
    for nombre, (n, mu, sd) in perfiles.items():
        df_p = pd.DataFrame(rng.normal(mu, sd, size=(n, len(cols))), columns=cols)
        df_p["perfil"] = nombre
        partes.append(df_p)

    # Clientes atípicos (valores extremos)
    n_out = 12
    atipicos = pd.DataFrame({
        "edad": rng.uniform(25, 60, n_out),
        "ingreso_anual": rng.uniform(200_000, 300_000, n_out),
        "gasto_promedio": rng.uniform(600, 900, n_out),
        "frecuencia_compras": rng.uniform(25, 40, n_out),
        "visitas_web_mes": rng.uniform(45, 70, n_out),
        "pct_descuento": rng.uniform(0, 1, n_out),
    })
    atipicos["perfil"] = "Atipico"
    partes.append(atipicos)

    df = pd.concat(partes, ignore_index=True)

    # Variables redundantes (correlacionadas con otras) -> útiles para PCA
    df["tiempo_sesion_min"] = 0.8 * df["visitas_web_mes"] + rng.normal(0, 2, len(df))
    df["gasto_anual"] = df["gasto_promedio"] * df["frecuencia_compras"] + rng.normal(0, 100, len(df))

    df["pct_descuento"] = df["pct_descuento"].clip(0, 1)
    num = df.select_dtypes("number").columns
    df[num] = df[num].clip(lower=0)
    return df


df = generar_clientes()
FEATURES = [c for c in df.columns if c != "perfil"]
print(df.head())
# df.to_csv("clientes.csv", index=False)

COLORES = {"Cazadores de ofertas": "tab:blue", "Compradores premium": "tab:orange",
           "Ocasionales": "tab:green", "Atipico": "tab:red"}
```

## 1. PCA

```python
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA().fit(X_scaled)
var = pca_full.explained_variance_ratio_
cum = np.cumsum(var)
n_comp_90 = int(np.argmax(cum >= 0.90) + 1)

# Demostración de la "regla de oro": PCA sin estandarizar
var_sin_escalar = PCA().fit(X).explained_variance_ratio_[0]
print(f"PC1 sin estandarizar: {var_sin_escalar:.1%} de la varianza (domina la variable de mayor escala)")
print(f"PC1 estandarizado:    {var[0]:.1%}")
print(f"Componentes para retener 90% de varianza: {n_comp_90} de {len(FEATURES)}")

X_2d = pca_full.transform(X_scaled)[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) Scree plot + varianza acumulada
axes[0].bar(range(1, len(var) + 1), var, alpha=0.7, label="Varianza individual")
axes[0].plot(range(1, len(var) + 1), cum, "o-", color="tab:red", label="Acumulada")
axes[0].axhline(0.90, ls="--", color="gray")
axes[0].set(xlabel="Componente principal", ylabel="Varianza explicada",
            title="PCA: varianza explicada")
axes[0].legend()

# (b) Proyección 2D coloreada por perfil real
for perfil, color in COLORES.items():
    m = df["perfil"] == perfil
    axes[1].scatter(X_2d[m, 0], X_2d[m, 1], s=18, alpha=0.7, c=color, label=perfil)
axes[1].set(xlabel=f"PC1 ({var[0]:.0%})", ylabel=f"PC2 ({var[1]:.0%})",
            title="Proyección 2D (8 variables -> 2 componentes)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## 2. K-means (sobre los componentes de PCA)

```python
pca90 = PCA(n_components=n_comp_90).fit(X_scaled)
X_pca = pca90.transform(X_scaled)

ks = range(2, 9)
inercias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(X_pca)
    inercias.append(km.inertia_)
    sils.append(silhouette_score(X_pca, km.labels_))

k_opt = list(ks)[int(np.argmax(sils))]
km = KMeans(n_clusters=k_opt, n_init=10, random_state=RNG).fit(X_pca)
print(f"K elegido por silhouette: {k_opt}")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

axes[0].plot(list(ks), inercias, "o-")
axes[0].set(xlabel="K", ylabel="Inercia", title="Método del codo")

axes[1].plot(list(ks), sils, "o-", color="tab:green")
axes[1].axvline(k_opt, ls="--", color="gray")
axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette Score")

sc = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap="viridis", s=18, alpha=0.8)
axes[2].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                c="red", marker="X", s=200, edgecolor="black", label="Centroides")
axes[2].set(xlabel="PC1", ylabel="PC2", title=f"K-means (K={k_opt})")
axes[2].legend()
plt.tight_layout()
plt.show()

# Perfilado: centroides en la escala original para poder "nombrar" cada cluster
centroides = scaler.inverse_transform(pca90.inverse_transform(km.cluster_centers_))
perfil_clusters = pd.DataFrame(centroides, columns=FEATURES).round(1)
perfil_clusters.index.name = "cluster"
print(perfil_clusters)
```

## 3. DBSCAN vs. K-means (formas irregulares + ruido)

```python
rng = np.random.default_rng(RNG)
X_moons, _ = make_moons(n_samples=500, noise=0.07, random_state=RNG)
ruido = rng.uniform([-1.5, -1.0], [2.5, 1.5], size=(25, 2))
X_m = np.vstack([X_moons, ruido])

lab_km = KMeans(n_clusters=2, n_init=10, random_state=RNG).fit_predict(X_m)
lab_db = DBSCAN(eps=0.15, min_samples=8).fit_predict(X_m)


def plot_clusters(ax, X, labels, titulo):
    for lab in np.unique(labels):
        m = labels == lab
        if lab == -1:
            ax.scatter(X[m, 0], X[m, 1], c="gray", marker="x", s=30, label="Ruido")
        else:
            ax.scatter(X[m, 0], X[m, 1], s=18, alpha=0.8, label=f"Cluster {lab}")
    ax.set_title(titulo)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_clusters(axes[0], X_m, lab_km, "K-means: corta las lunas por la mitad")
plot_clusters(axes[1], X_m, lab_db, "DBSCAN: sigue la forma y aísla el ruido")
plt.tight_layout()
plt.show()
```

## 4. HDBSCAN vs. DBSCAN (densidades distintas)

```python
X_v, _ = make_blobs(n_samples=[300, 300, 300],
                    centers=[(-5, -5), (0, 5), (8, -2)],
                    cluster_std=[0.3, 0.8, 2.0], random_state=RNG)

lab_db2 = DBSCAN(eps=0.5, min_samples=10).fit_predict(X_v)
lab_hdb = HDBSCAN(min_cluster_size=30).fit_predict(X_v)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_clusters(axes[0], X_v, lab_db2, "DBSCAN (un solo eps para todos)")
plot_clusters(axes[1], X_v, lab_hdb, "HDBSCAN (eps adaptativo)")
plt.tight_layout()
plt.show()
```

Con `eps=0.5`, DBSCAN debería capturar bien el grupo denso, pero el grupo más disperso queda casi entero marcado como ruido, porque sus puntos no alcanzan `min_samples` vecinos dentro de ese radio. Si subes `eps` para rescatarlo, el riesgo es fusionar los grupos densos. HDBSCAN evita ese dilema.

## Ventaja de cada técnica sobre las otras

| Técnica | Ventaja principal | Frente a quién | Limitación |
| --- | --- | --- | --- |
| **PCA** | Comprime variables correlacionadas conservando la mayor parte de la varianza; acelera y estabiliza el clustering y permite visualizar en 2D. | Frente a la selección de características: aprovecha la información de todas las variables en lugar de descartarlas. | Pierde interpretabilidad directa y solo captura relaciones lineales. |
| **K-means** | Simple, rápido, escalable y produce centroides fáciles de perfilar y explicar al negocio. | Frente a DBSCAN/HDBSCAN: escala mejor a datasets grandes y siempre asigna todos los puntos a un grupo. | Hay que fijar K, asume grupos esféricos y de tamaño similar, y es sensible a outliers. |
| **DBSCAN** | No requiere K, encuentra formas arbitrarias y detecta ruido de forma nativa. | Frente a K-means: separa formas no convexas (lunas, anillos) y no se distorsiona por outliers. | Un único `eps` no sirve cuando las densidades varían; es sensible a `eps` y `min_samples`. |
| **HDBSCAN** | Maneja densidades variables, requiere ajustar menos parámetros (básicamente `min_cluster_size`) y es más robusto. | Frente a DBSCAN: no depende de un `eps` global. | Más costoso computacionalmente y los resultados son menos directos de explicar que un centroide. |

## Criterio práctico

- Si necesitas **segmentos explicables para negocio** y los grupos son razonablemente compactos, usa PCA y luego K-means, tal como lo plantea el pipeline del documento.
- Si hay **mucho ruido o formas raras**, pasa a DBSCAN.
- Si además las **densidades son muy distintas** entre grupos, usa HDBSCAN.

Un detalle útil para el material del curso: en la sección 1, la comparación entre "PC1 sin estandarizar" y "PC1 estandarizado" ilustra de forma numérica la regla de oro del documento (estandarizar antes de PCA), con la variable `ingreso_anual` dominando por su escala.



---

Hay una forma directa de medirlo: en cada paso, la varianza que descartas es la información que pierdes. Los snippets siguientes reutilizan `X_scaled`, `var`, `cum`, `pca_full` y `n_comp_90` del script anterior.

## 1. PCA: varianza retenida vs. perdida

Como `X_scaled` está centrado, la fracción de error de reconstrucción es exactamente `1 - varianza acumulada`. El código lo verifica reconstruyendo los datos con `inverse_transform`.

```python
ks_pca = range(1, X_scaled.shape[1] + 1)
perdida_teorica, perdida_real = [], []

for k in ks_pca:
    p = PCA(n_components=k).fit(X_scaled)
    X_rec = p.inverse_transform(p.transform(X_scaled))
    error = np.sum((X_scaled - X_rec) ** 2) / np.sum(X_scaled ** 2)
    perdida_real.append(error)
    perdida_teorica.append(1 - cum[k - 1])

print(f"Verificación (k={n_comp_90}): perdida real = {perdida_real[n_comp_90-1]:.4f} "
      f"| 1 - varianza acumulada = {perdida_teorica[n_comp_90-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) Retenida vs. perdida según nº de componentes
axes[0].stackplot(list(ks_pca),
                  [cum, 1 - cum],
                  labels=["Retenida", "Perdida"],
                  colors=["tab:blue", "tab:red"], alpha=0.6)
axes[0].axvline(n_comp_90, ls="--", color="black")
axes[0].set(xlabel="Nº de componentes", ylabel="Proporción de varianza",
            title="PCA: información retenida vs. perdida", ylim=(0, 1))
axes[0].legend(loc="center right")

# (b) Cuánto se pierde de cada variable original (R² de reconstrucción)
def r2_por_variable(k):
    p = PCA(n_components=k).fit(X_scaled)
    X_rec = p.inverse_transform(p.transform(X_scaled))
    return 1 - np.sum((X_scaled - X_rec) ** 2, axis=0) / np.sum(X_scaled ** 2, axis=0)

ancho = 0.38
idx = np.arange(len(FEATURES))
axes[1].bar(idx - ancho/2, r2_por_variable(2), ancho, label="2 componentes")
axes[1].bar(idx + ancho/2, r2_por_variable(n_comp_90), ancho, label=f"{n_comp_90} componentes")
axes[1].set_xticks(idx)
axes[1].set_xticklabels(FEATURES, rotation=45, ha="right")
axes[1].set(ylabel="Varianza conservada de la variable", title="Pérdida por variable original")
axes[1].legend()
plt.tight_layout()
plt.show()
```

El gráfico (b) aporta un matiz que el porcentaje global esconde: la pérdida no se reparte de forma pareja. Una variable que no está correlacionada con las demás (por ejemplo `edad` o `pct_descuento`) puede quedar mal representada aunque la varianza total retenida sea alta.

## 2. K-means: varianza explicada por los clusters

Al reducir cada cliente a "su centroide" también se pierde detalle. Esa pérdida es la inercia (varianza dentro de los grupos), y la proporción explicada es la misma idea que el R² de un ANOVA:

```python
SST = np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2)   # varianza total

ks_km = range(1, 11)
explicada_km = []
for k in ks_km:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(X_scaled)
    explicada_km.append(1 - km_k.inertia_ / SST)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(list(ks_km), explicada_km, "o-")
axes[0].set(xlabel="K", ylabel="Varianza explicada (entre grupos)",
            title="K-means: 1 - inercia / varianza total", ylim=(0, 1))

# Pipeline completo: PCA + K-means respecto de la información ORIGINAL
SST_pca = np.sum((X_pca - X_pca.mean(axis=0)) ** 2)
retenida_pca = SST_pca / SST
explicada_total = [retenida_pca * (1 - KMeans(n_clusters=k, n_init=10, random_state=RNG)
                                   .fit(X_pca).inertia_ / SST_pca) for k in ks_km]

axes[1].plot(list(ks_km), explicada_km, "o-", label="K-means sin PCA")
axes[1].plot(list(ks_km), explicada_total, "s--", label=f"PCA ({n_comp_90} comp.) + K-means")
axes[1].set(xlabel="K", ylabel="Varianza original explicada",
            title="Costo acumulado de encadenar reducciones", ylim=(0, 1))
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"PCA retiene {retenida_pca:.1%}; con K={k_opt}, el pipeline explica "
      f"{explicada_total[k_opt-1]:.1%} de la varianza original")
```

La curva de la izquierda es otra lectura del método del codo: el codo marca el punto donde agregar un cluster más ya casi no recupera varianza. En el gráfico de la derecha, la curva con PCA queda siempre por debajo porque las pérdidas se multiplican: lo que descartó PCA no lo recupera K-means.

## 3. DBSCAN y HDBSCAN

No tienen centroides ni inercia, así que no existe una "varianza perdida" única. Se puede calcular la misma medida usando la media de cada cluster, pero con dos precauciones:

```python
def varianza_explicada(X, labels):
    mask = labels != -1                       # se excluye el ruido
    Xc, lc = X[mask], labels[mask]
    sst = np.sum((Xc - Xc.mean(axis=0)) ** 2)
    ssw = sum(np.sum((Xc[lc == l] - Xc[lc == l].mean(axis=0)) ** 2) for l in np.unique(lc))
    return 1 - ssw / sst, 1 - mask.mean()     # (varianza explicada, fracción de ruido)

for nombre, lab in [("K-means", lab_km), ("DBSCAN", lab_db)]:
    r2, ruido_frac = varianza_explicada(X_m, lab)
    print(f"{nombre}: varianza explicada = {r2:.1%} | puntos como ruido = {ruido_frac:.1%}")
```

Las precauciones son estas:

- **El ruido no entra en el cálculo.** DBSCAN y HDBSCAN "descartan" puntos, y esa es una pérdida distinta, de observaciones completas, por lo que conviene reportar siempre la fracción de ruido junto con la varianza explicada.
- **Una varianza explicada más baja no significa peor resultado.** En las lunas, K-means probablemente tenga mayor varianza explicada que DBSCAN, pero sus grupos están mal formados. Esa métrica supone grupos compactos y esféricos, y solo es una buena vara para K-means. Para formas arbitrarias conviene evaluar con otro criterio, como un índice basado en densidad (DBCV) o la validación visual.

## Resumen

| Técnica | Qué se pierde | Cómo se mide |
| --- | --- | --- |
| PCA | Varianza de los componentes descartados | `1 - explained_variance_ratio_.sum()`, o error de reconstrucción |
| K-means | Variación individual dentro de cada grupo | Inercia / varianza total |
| DBSCAN / HDBSCAN | Puntos marcados como ruido y detalle intra-cluster | Fracción de ruido + varianza intra-cluster sin ruido |

Igual que con el script anterior, no ejecuté este código, así que los porcentajes concretos dependerán de tu dataset generado.

---

No existe un K "óptimo" único. Lo recomendable es calcular varias métricas, ver dónde coinciden y validar el resultado con criterio de negocio. Este código reutiliza `X_pca`, `X_scaled` y `RNG` del script anterior.

## 1. Varias métricas en una sola pasada

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

ks = list(range(2, 11))
res = {"inercia": [], "silhouette": [], "calinski": [], "davies": [], "bic": []}

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(X_pca)
    res["inercia"].append(km.inertia_)
    res["silhouette"].append(silhouette_score(X_pca, km.labels_))
    res["calinski"].append(calinski_harabasz_score(X_pca, km.labels_))
    res["davies"].append(davies_bouldin_score(X_pca, km.labels_))
    res["bic"].append(GaussianMixture(k, n_init=3, random_state=RNG).fit(X_pca).bic(X_pca))


def codo_automatico(ks, inercias):
    """Punto de la curva más alejado de la recta que une el primer y el último punto."""
    x = np.array(ks, float); y = np.array(inercias, float)
    x, y = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return ks[int(np.argmax(dist))]


mejores = {
    "Codo": codo_automatico(ks, res["inercia"]),
    "Silhouette (max)": ks[int(np.argmax(res["silhouette"]))],
    "Calinski-Harabasz (max)": ks[int(np.argmax(res["calinski"]))],
    "Davies-Bouldin (min)": ks[int(np.argmin(res["davies"]))],
    "BIC / GMM (min)": ks[int(np.argmin(res["bic"]))],
}
print(pd.Series(mejores, name="K sugerido"))

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
config = [("inercia", "Codo (inercia)", "tab:blue"),
          ("silhouette", "Silhouette (maximizar)", "tab:green"),
          ("calinski", "Calinski-Harabasz (maximizar)", "tab:orange"),
          ("davies", "Davies-Bouldin (minimizar)", "tab:red"),
          ("bic", "BIC de GMM (minimizar)", "tab:purple")]
for ax, (clave, titulo, color) in zip(axes, config):
    ax.plot(ks, res[clave], "o-", color=color)
    ax.set(xlabel="K", title=titulo)
plt.tight_layout()
plt.show()
```

Cómo leer cada una:

| Métrica | Qué mide | Criterio |
| --- | --- | --- |
| Codo (inercia) | Compacidad interna de los grupos | Donde la curva deja de bajar rápido |
| Silhouette | Cohesión vs. separación, por punto (-1 a 1) | Máximo |
| Calinski-Harabasz | Varianza entre grupos / varianza dentro | Máximo |
| Davies-Bouldin | Similitud promedio entre cada grupo y su vecino más parecido | Mínimo |
| BIC (GMM) | Ajuste probabilístico penalizando complejidad | Mínimo |

## 2. Silhouette detallado por cluster

El promedio puede esconder un cluster malo. Este gráfico muestra la silueta de cada punto agrupada por cluster:

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

candidatos = [3, 4, 5]
fig, axes = plt.subplots(1, len(candidatos), figsize=(15, 4.5))

for ax, k in zip(axes, candidatos):
    labels = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit_predict(X_pca)
    s = silhouette_samples(X_pca, labels)
    y_ini = 10
    for c in range(k):
        s_c = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(y_ini, y_ini + len(s_c)), 0, s_c,
                         color=cm.viridis(c / k), alpha=0.8)
        ax.text(-0.05, y_ini + len(s_c) / 2, str(c))
        y_ini += len(s_c) + 10
    ax.axvline(s.mean(), color="red", ls="--")
    ax.set(title=f"K={k} (promedio {s.mean():.2f})", xlabel="Silhouette", yticks=[])
plt.tight_layout()
plt.show()
```

Buscas que todos los clusters superen la línea roja del promedio, que tengan un grosor parecido y que no tengan valores negativos (puntos probablemente mal asignados).

## 3. Gap statistic (opcional, más robusto)

Compara tu inercia con la que se obtendría con datos sin estructura (uniformes). Es más lento, pero evita el sesgo de las otras métricas hacia K pequeños:

```python
def gap_statistic(X, ks, B=10, seed=RNG):
    rng = np.random.default_rng(seed)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    gaps, sds = [], []
    for k in ks:
        log_w = np.log(KMeans(k, n_init=10, random_state=seed).fit(X).inertia_)
        log_ref = np.array([
            np.log(KMeans(k, n_init=10, random_state=seed)
                   .fit(rng.uniform(mins, maxs, X.shape)).inertia_)
            for _ in range(B)])
        gaps.append(log_ref.mean() - log_w)
        sds.append(log_ref.std() * np.sqrt(1 + 1 / B))
    return np.array(gaps), np.array(sds)


gaps, sds = gap_statistic(X_pca, ks)
# Regla de Tibshirani: el menor K tal que gap(k) >= gap(k+1) - sd(k+1)
k_gap = next((ks[i] for i in range(len(ks) - 1) if gaps[i] >= gaps[i + 1] - sds[i + 1]), ks[-1])

plt.errorbar(ks, gaps, yerr=sds, fmt="o-", capsize=3)
plt.axvline(k_gap, ls="--", color="gray")
plt.xlabel("K"); plt.ylabel("Gap"); plt.title(f"Gap statistic (K sugerido = {k_gap})")
plt.show()
```

## 4. Estabilidad (opcional)

Un K es confiable si al remuestrear los datos aparecen los mismos grupos:

```python
from sklearn.metrics import adjusted_rand_score

def estabilidad(X, k, n_rep=20, frac=0.8, seed=RNG):
    rng = np.random.default_rng(seed)
    base = KMeans(k, n_init=10, random_state=seed).fit(X)
    scores = []
    for _ in range(n_rep):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        sub = KMeans(k, n_init=10, random_state=int(rng.integers(1e6))).fit(X[idx])
        scores.append(adjusted_rand_score(base.predict(X[idx]), sub.labels_))
    return np.mean(scores)

for k in ks:
    print(f"K={k}: estabilidad (ARI) = {estabilidad(X_pca, k):.2f}")
```

Valores cercanos a 1 indican una partición estable.

## Cómo decidir en la práctica

1. Busca el K en el que coincidan la mayoría de las métricas. Si divergen, prioriza silhouette y estabilidad.
2. Desconfía de K=2: silhouette y Calinski-Harabasz suelen favorecerlo porque separar "grande vs. chico" casi siempre da buena separación, aunque haya subgrupos reales. En tu dataset hay 3 perfiles y un grupo de atípicos, así que es probable que las métricas se repartan entre 2, 3 y 4, y que el outlier-cluster distorsione algo.
3. Perfila los centroides (como en el script anterior) para los 2 o 3 K candidatos y elige el que produzca segmentos distintos, explicables y accionables. Es el criterio de negocio del documento, y a veces decide entre dos K con métricas casi iguales.

## Para DBSCAN y HDBSCAN

No eligen K, pero sí tienen parámetros que ajustar:

```python
from sklearn.neighbors import NearestNeighbors

k_vecinos = 8   # igual a min_samples
dist, _ = NearestNeighbors(n_neighbors=k_vecinos).fit(X_m).kneighbors(X_m)
plt.plot(np.sort(dist[:, -1]))
plt.xlabel("Puntos ordenados"); plt.ylabel(f"Distancia al vecino nº {k_vecinos}")
plt.title("k-distance plot: el codo sugiere eps")
plt.show()
```

El codo de esa curva es un buen `eps` para DBSCAN. Para HDBSCAN, el parámetro clave es `min_cluster_size`, que se define por el tamaño mínimo de grupo que tiene sentido para el negocio.

No ejecuté este código, así que los K sugeridos dependerán del dataset generado. El gap statistic y la estabilidad pueden tardar un poco por la cantidad de ajustes de K-means.
