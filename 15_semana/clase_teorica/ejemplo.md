Usa el dataset real. Es más rico que uno sintético y ya tienes la URL. Adapté el ejercicio de ensambles al caso Airbnb, con el target binario `inactivo` (`number_of_reviews_ltm == 0`), en celdas de notebook con markdown y código.

No puedo ejecutar código en esta conversación, así que no te doy cifras inventadas. Al final dejo las hipótesis que debes verificar con los resultados, y si me pegas los outputs redacto las conclusiones con los números reales.

---

## Ejercicio: comparación de ensambles para predecir listings inactivos en Airbnb Buenos Aires

**Contexto.** Una plataforma quiere detectar alojamientos sin actividad en los últimos 12 meses, para contactar a los anfitriones o depurar el catálogo. Un listing es *inactivo* si `number_of_reviews_ltm == 0`.

**Objetivo del alumno.**
1. Hacer un análisis estadístico exploratorio que justifique las decisiones de modelado.
2. Comparar Bagging, AdaBoost y Gradient Boosting con validación cruzada solo sobre train.
3. Optimizar el mejor candidato con `RandomizedSearchCV`.
4. Elegir el umbral de decisión con predicciones out-of-fold, no con test.
5. Evaluar una única vez sobre test e interpretar el resultado.

---

## Parte 1. Carga y calidad de datos

```python
import numpy as np
import pandas as pd
from scipy import stats

URL = "https://raw.githubusercontent.com/Andru-1987/csv_files_ds/refs/heads/main/buenos_aires.csv"
df = pd.read_csv(URL)

print(df.shape)
calidad = pd.DataFrame({
    "nulos": df.isna().sum(),
    "pct_nulos": (df.isna().mean() * 100).round(1),
    "unicos": df.nunique(),
})
calidad.sort_values("pct_nulos", ascending=False)
```

**Qué mirar.** En la muestra que compartiste, `neighbourhood_group` y `license` vienen casi vacías, `price` tiene nulos y `reviews_per_month` y `last_review` son nulos cuando no hay reseñas. Los dos últimos son consecuencia directa de no tener reseñas, y eso importa para el leakage (Parte 4).

```python
df = df.drop_duplicates("id")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0].copy()

df["inactivo"] = (df["number_of_reviews_ltm"] == 0).astype(int)
df["log_price"] = np.log1p(df["price"])

print(f"Filas tras limpieza: {len(df)}")
print(f"Proporción de inactivos: {df['inactivo'].mean():.3f}")
```

**Pregunta para el alumno.** ¿Qué tan desbalanceado está el target? Esa proporción decide si accuracy sirve como métrica y qué baseline usar.

---

## Parte 2. Análisis estadístico univariado

```python
# Forma de la distribución de price
print(df["price"].describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(0))
print(f"Asimetría price:     {df['price'].skew():.2f}")
print(f"Asimetría log_price: {df['log_price'].skew():.2f}")
print(f"Curtosis price:      {df['price'].kurt():.2f}")

# Outliers por IQR
q1, q3 = df["price"].quantile([.25, .75])
iqr = q3 - q1
print(f"% outliers superiores (IQR): {(df['price'] > q3 + 1.5 * iqr).mean() * 100:.1f}%")

# Estadías largas (alquiler mensual encubierto)
print(f"% con minimum_nights >= 28: {(df['minimum_nights'] >= 28).mean() * 100:.1f}%")

# Composición
print(df["room_type"].value_counts(normalize=True).round(3))
```

**Para interpretar.** Con asimetría alta, la media es engañosa y conviene reportar la mediana y trabajar con `log_price`. El `price` está en pesos argentinos, así que la dispersión también refleja el contexto macroeconómico. Los árboles son invariantes a transformaciones monótonas, pero la transformación sí importa para la visualización y para la regresión lineal.

---

## Parte 3. Análisis bivariado con tests de hipótesis

Como `price` no es normal, se usan tests no paramétricos.

```python
# 3.1 Precio por tipo de alojamiento: Kruskal-Wallis (>2 grupos)
grupos = [g["price"].values for _, g in df.groupby("room_type")]
H, p = stats.kruskal(*grupos)
print(f"Kruskal-Wallis price ~ room_type: H={H:.1f}, p={p:.2e}")
print(df.groupby("room_type")["price"].agg(["count", "median"]).round(0))

# 3.2 Precio: inactivos vs activos (Mann-Whitney + tamaño de efecto)
a = df.loc[df["inactivo"] == 1, "price"]
b = df.loc[df["inactivo"] == 0, "price"]
U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
r_rb = 2 * U / (len(a) * len(b)) - 1   # rank-biserial, rango [-1, 1]
print(f"Mann-Whitney: p={p:.2e}, rank-biserial={r_rb:.3f}")

# 3.3 Inactividad vs tipo de alojamiento: chi-cuadrado + V de Cramer
tabla = pd.crosstab(df["room_type"], df["inactivo"])
chi2, p, dof, _ = stats.chi2_contingency(tabla)
v = np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))
print(f"Chi2 inactivo ~ room_type: p={p:.2e}, V de Cramer={v:.3f}")

# 3.4 Correlación de Spearman entre numéricas
NUMS = ["price", "minimum_nights", "number_of_reviews",
        "calculated_host_listings_count", "availability_365", "number_of_reviews_ltm"]
print(df[NUMS].corr(method="spearman").round(2))

# 3.5 Barrios: precio mediano y tasa de inactividad (barrios con n >= 30)
barrios = (df.groupby("neighbourhood")
             .agg(n=("id", "count"),
                  precio_mediano=("price", "median"),
                  tasa_inactivo=("inactivo", "mean"))
             .query("n >= 30")
             .sort_values("precio_mediano", ascending=False))
print(barrios.head(10).round(2))
print(barrios.tail(10).round(2))

# 3.6 Profesionalización: hosts con varios listings vs uno solo
print(df.groupby(df["calculated_host_listings_count"] > 1)["inactivo"].mean().round(3))
```

**Cómo leer los resultados.**
- Con muestras grandes, casi todo test da p muy pequeño. Por eso se reporta siempre el tamaño de efecto (rank-biserial, V de Cramer), no solo el p-valor.
- Una correlación de Spearman alta entre `number_of_reviews_ltm` y `number_of_reviews` anticipa el problema de leakage.

---

## Parte 4. Preparación para modelar y decisión anti-leakage

Decisión de diseño que el alumno debe justificar:

| Variable | Decisión | Motivo |
| --- | --- | --- |
| `number_of_reviews_ltm` | Excluir | Es el target mismo. |
| `reviews_per_month` | Excluir | Se calcula a partir de la actividad reciente y es nulo exactamente cuando no hay reseñas. |
| `last_review` | Excluir | Revela directamente si hubo reseñas recientes. |
| `id`, `host_id`, `host_profile_id`, `name`, `host_name` | Excluir | Identificadores o texto libre. |
| `number_of_reviews` | Incluir, con discusión | Es histórico y no contiene el target, pero está correlacionado con él. |

**Confusor a discutir.** Un listing nuevo sin reseñas aparece como inactivo sin serlo realmente. El alumno puede evaluar el modelo con y sin `number_of_reviews` para medir cuánto depende de esa variable.

```python
from sklearn.model_selection import train_test_split

NUM = ["price", "minimum_nights", "number_of_reviews",
       "calculated_host_listings_count", "availability_365", "latitude", "longitude"]
CAT = ["room_type", "neighbourhood"]

X, y = df[NUM + CAT], df["inactivo"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Inactivos Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")
```

---

## Parte 5. Baselines por validación cruzada (solo train)

Esta versión corrige un punto del ejercicio de crédito: allí los baselines se evaluaban sobre test, lo que contamina la selección de modelo. Aquí todo se compara con CV sobre train.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_validate

def make_pipe(model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "Dummy (prior)": DummyClassifier(strategy="prior"),
    "Random Forest (Bagging)": RandomForestClassifier(random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

filas = []
for nombre, m in modelos.items():
    r = cross_validate(make_pipe(m), X_train, y_train, cv=cv,
                       scoring=["roc_auc", "average_precision"])
    filas.append({
        "modelo": nombre,
        "auc_media": r["test_roc_auc"].mean(),
        "auc_std": r["test_roc_auc"].std(),
        "pr_auc_media": r["test_average_precision"].mean(),
    })

resultados = pd.DataFrame(filas).sort_values("auc_media", ascending=False).round(4)
resultados
```

**Para interpretar.**
- El Dummy da AUC cercano a 0,5 y un PR-AUC igual a la proporción de inactivos. Cualquier modelo útil debe superar esos pisos con claridad.
- Si dos modelos difieren en menos de un desvío estándar (`auc_std`), la diferencia no es concluyente y conviene elegir el más simple o barato.

---

## Parte 6. Optimización con RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

DISTS = {
    "Random Forest (Bagging)": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt", 0.5],
    },
    "AdaBoost": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.7, 0.85, 1.0],
    },
}

ganador = resultados[resultados["modelo"] != "Dummy (prior)"].iloc[0]["modelo"]
print("Modelo a optimizar:", ganador)

busqueda = RandomizedSearchCV(
    make_pipe(modelos[ganador]),
    param_distributions=DISTS[ganador],
    n_iter=12,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
)
busqueda.fit(X_train, y_train)   # solo train
best = busqueda.best_estimator_

print("Mejores hiperparámetros:", busqueda.best_params_)
print(f"AUC CV del mejor: {busqueda.best_score_:.4f}")
```

---

## Parte 7. Umbral de decisión con predicciones out-of-fold

El umbral por defecto (0,5) rara vez es el adecuado con clases desbalanceadas. Se elige con predicciones out-of-fold sobre train, nunca sobre test.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

MIN_PREC = 0.60   # precisión mínima exigida (regla de negocio)

oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)

ok = prec[:-1] >= MIN_PREC
if ok.any():
    umbral = thr[ok].min()          # menor umbral que cumple la precisión => máximo recall
else:
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    umbral = thr[np.argmax(f1)]
    print("Ningún umbral alcanza la precisión mínima; se usa el de mayor F1.")

print(f"Umbral elegido: {umbral:.3f}")
```

---

## Parte 8. Evaluación final en test (una sola vez)

```python
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
from sklearn.inspection import permutation_importance

y_prob = best.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= umbral).astype(int)

print(f"AUC-ROC test: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC test:  {average_precision_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))

# Importancia por permutación (sobre columnas originales, gracias al Pipeline)
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                             n_repeats=5, random_state=42, n_jobs=-1)
(pd.Series(imp.importances_mean, index=X_test.columns)
   .sort_values(ascending=False).round(4))
```

---

## Hipótesis a verificar con tus resultados

Son hipótesis razonables para este tipo de datos, no resultados.

| Hipótesis | Cómo verificarla |
| --- | --- |
| `price` es muy asimétrico y `log_price` corrige gran parte de la asimetría. | Parte 2, asimetría antes y después. |
| El tipo de alojamiento y el barrio explican diferencias de precio con efecto grande. | Kruskal-Wallis y tabla de barrios. |
| Los inactivos tienen menos `number_of_reviews` históricas y más `availability_365`. | Spearman y permutation importance. |
| Los hosts con un único listing tienen una tasa de inactividad distinta a los profesionales. | Parte 3.6. |
| Los ensambles superan claramente al Dummy en AUC y PR-AUC. | Parte 5. |
| Las diferencias entre los tres ensambles son pequeñas (menos de un desvío estándar). | Columna `auc_std`. |
| Quitar `number_of_reviews` baja el AUC de forma notable. | Repetir las Partes 5 a 8 sin esa columna. |

---

## Rúbrica

1. **Leakage:** el tuning y la elección de umbral usan solo train. Se penaliza el uso de `number_of_reviews_ltm`, `reviews_per_month` o `last_review` como features.
2. **Métricas:** se justifica AUC-ROC y PR-AUC frente a accuracy según la proporción real de inactivos.
3. **Estadística:** se usan tests no paramétricos con tamaño de efecto, y se explica por qué no se usa t-test.
4. **Costos:** el alumno argumenta qué error pesa más (no detectar un inactivo frente a contactar a un host activo) y cómo eso justifica `MIN_PREC`.
5. **Confusor:** se discute el caso de los listings nuevos sin reseñas y se prueba el modelo sin `number_of_reviews`.

Si ejecutas las celdas y me pegas las tablas de las Partes 1 a 3, la Parte 5 y la matriz de confusión final, redacto la sección de conclusiones con los números reales. Si prefieres que el target sea regresión sobre `price`, también puedo adaptar el pipeline.
