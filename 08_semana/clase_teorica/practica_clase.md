### 1. Preparación y Generación del Dataset Sintético

Primero, importamos las librerías estándar de la industria y creamos un dataset diseñado específicamente para mostrar asimetría (skewness), correlaciones no lineales y diferencias entre grupos.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Configuración visual
sns.set_theme(style="whitegrid")
np.random.seed(42)

# Generación de 1000 empleados
n_empleados = 1000

# 1. Años de experiencia (Normalmente distribuida, limitada a positivos)
experiencia = np.clip(np.random.normal(loc=10, scale=5, size=n_empleados), 0, 40)

# 2. Salario (Con asimetría positiva fuerte y relación exponencial con experiencia)
# Simulamos que el salario crece de forma no lineal (exponencial) con la experiencia,
# y le sumamos ruido exponencial para simular outliers (sueldos de directivos).
salario_base = 30000 * np.exp(0.08 * experiencia)
ruido_salario = np.random.exponential(scale=20000, size=n_empleados) 
salario = salario_base + ruido_salario

# 3. Nivel de Desempeño (Ordinal: 1 a 5)
desempenio = np.clip(np.round(np.random.normal(3, 1, n_empleados)), 1, 5)

# 4. Departamento (Categórica)
departamentos = np.random.choice(['IT', 'Ventas', 'RRHH', 'Marketing'], size=n_empleados, p=[0.4, 0.3, 0.15, 0.15])

# 5. Promocionado (Categórica: Sí/No, depende del desempeño)
prob_promocion = desempenio / 10  # A mayor desempeño, más probabilidad
promocionado = np.random.binomial(1, prob_promocion)

# Creación del DataFrame
df = pd.DataFrame({
    'experiencia_anios': experiencia,
    'salario_usd': salario,
    'nivel_desempenio': desempenio,
    'departamento': departamentos,
    'promocionado': np.where(promocionado == 1, 'Si', 'No')
})

df.head()

```

---

### 2. Análisis Univariado y Métricas Estadísticas (Sección 8.1 y 8.3)

Vamos a analizar la variable `salario_usd`. Calcularemos las medidas de tendencia central, dispersión, asimetría y curtosis, y luego las visualizaremos.

```python
# --- 1. Cálculo de Métricas ---
media = df['salario_usd'].mean()
mediana = df['salario_usd'].median()
desvio_std = df['salario_usd'].std()

# Métricas de forma (Shape)
skewness = stats.skew(df['salario_usd'])
kurtosis = stats.kurtosis(df['salario_usd']) # Devuelve el exceso de curtosis por defecto

print("--- ANÁLISIS UNIVARIADO: SALARIO ---")
print(f"Media: ${media:,.2f}")
print(f"Mediana: ${mediana:,.2f}")
print(f"Desviación Estándar: ${desvio_std:,.2f}")
print(f"Asimetría (Skewness): {skewness:.2f}")
print(f"Curtosis (Exceso): {kurtosis:.2f}")

# --- 2. Visualización: Histograma y Boxplot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma con curva de densidad (KDE)
sns.histplot(df['salario_usd'], kde=True, ax=axes[0], color='blue', bins=40)
axes[0].axvline(media, color='red', linestyle='--', label=f'Media: {media/1000:.1f}k')
axes[0].axvline(mediana, color='green', linestyle='-', label=f'Mediana: {mediana/1000:.1f}k')
axes[0].set_title('Distribución de Salarios')
axes[0].legend()

# Boxplot
sns.boxplot(x=df['salario_usd'], ax=axes[1], color='lightblue')
axes[1].set_title('Detección de Outliers (Boxplot)')

plt.tight_layout()
plt.show()

```

**Análisis de Resultados (Qué buscar en el output):**

* **Skewness > 1:** Vas a notar que la media es considerablemente mayor que la mediana. El gráfico mostrará una "cola" larga hacia la derecha.
* **Curtosis > 0 (Leptocúrtica):** El histograma tendrá un pico muy pronunciado y el boxplot mostrará una gran cantidad de puntos (outliers) a la derecha del "bigote" superior.

---

### 3. Análisis Bivariado: Numérico vs Numérico y Correlaciones (Sección 8.2 y 8.3)

Aquí evaluamos la relación entre `experiencia_anios` y `salario_usd` utilizando Pearson, Spearman y Kendall para comparar cómo reaccionan a relaciones no lineales y outliers.

```python
# --- 1. Cálculo de Correlaciones ---
# Pearson: Mide relación lineal
pearson_r, p_p = stats.pearsonr(df['experiencia_anios'], df['salario_usd'])

# Spearman: Mide relación monótona (robusta ante la relación exponencial y outliers)
spearman_r, p_s = stats.spearmanr(df['experiencia_anios'], df['salario_usd'])

# Kendall: Concordancia de rangos (usamos Nivel de Desempeño vs Experiencia para este ejemplo)
kendall_t, p_k = stats.kendalltau(df['nivel_desempenio'], df['salario_usd'])

print("\n--- ANÁLISIS BIVARIADO: CORRELACIONES ---")
print(f"Pearson (Experiencia vs Salario): {pearson_r:.3f}")
print(f"Spearman (Experiencia vs Salario): {spearman_r:.3f}")
print(f"Kendall Tau (Desempeño vs Salario): {kendall_t:.3f}")

# --- 2. Visualización: Scatter plot con línea de tendencia ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='experiencia_anios', y='salario_usd', alpha=0.5)
# Añadimos una línea de regresión (Lowess para capturar la curva)
sns.regplot(data=df, x='experiencia_anios', y='salario_usd', scatter=False, color='red', lowess=True)
plt.title('Relación: Experiencia vs Salario (Observa la curva exponencial)')
plt.show()

```

**Análisis de Resultados:**

* Verás que **Spearman > Pearson**. Esto se debe a que generamos los datos con una relación *exponencial* (monótona, pero no una línea recta perfecta). Pearson penaliza esa curva, Spearman no.

---

### 4. Análisis Bivariado: Categórico vs Numérico y Categórico vs Categórico

Veamos cómo comparar distribuciones entre grupos (ANOVA) y dependencias entre categorías (Chi-Cuadrado).

```python
# --- 1. Categórico vs Numérico: Salario por Departamento (ANOVA) ---
plt.figure(figsize=(10, 5))
# Ordenamos el boxplot por la mediana para facilitar la lectura
orden_deptos = df.groupby('departamento')['salario_usd'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='departamento', y='salario_usd', order=orden_deptos, palette='Set2')
plt.title('Salarios por Departamento')
plt.show()

# Test ANOVA de una vía
grupos_salario = [df[df['departamento'] == dpto]['salario_usd'] for dpto in df['departamento'].unique()]
f_stat, p_valor_anova = stats.f_oneway(*grupos_salario)
print(f"\nANOVA (Salario según Departamento): P-Valor = {p_valor_anova:.4f}")
# Si p_valor_anova > 0.05, no hay diferencia significativa real entre departamentos.


# --- 2. Categórico vs Categórico: Promoción por Departamento (Chi-Cuadrado) ---
# Creamos la tabla de contingencia
tabla_contingencia = pd.crosstab(df['departamento'], df['promocionado'])
print("\nTabla de Contingencia (Cantidades reales):")
print(tabla_contingencia)

# Test Chi-Cuadrado
chi2, p_valor_chi, dof, esperados = stats.chi2_contingency(tabla_contingencia)
print(f"\nTest Chi-Cuadrado (Independencia Depto vs Promoción): P-Valor = {p_valor_chi:.4f}")

# Visualización de proporciones (100% Stacked Bar Chart)
tabla_porcentual = pd.crosstab(df['departamento'], df['promocionado'], normalize='index') * 100
tabla_porcentual.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='viridis')
plt.title('Porcentaje de Promoción por Departamento')
plt.ylabel('Porcentaje %')
plt.legend(title='Promocionado', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

---

### 5. Tip en Acción (Sección 8.4): Transformación Logarítmica para Skewness

En Data Science, antes de entrenar un modelo predictivo, solemos corregir variables muy asimétricas. La transformación logarítmica es la más común.

```python
# Aplicamos np.log1p (logaritmo natural de 1 + x, útil si hay ceros)
df['log_salario'] = np.log1p(df['salario_usd'])

skewness_original = stats.skew(df['salario_usd'])
skewness_log = stats.skew(df['log_salario'])

print("\n--- CORRECCIÓN DE ASIMETRÍA ---")
print(f"Skewness Original: {skewness_original:.2f}")
print(f"Skewness post-Transformación: {skewness_log:.2f}")

# Comprobación visual del cambio
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(df['salario_usd'], kde=True, ax=axes[0], color='gray')
axes[0].set_title(f'Original (Skew: {skewness_original:.2f})')

sns.histplot(df['log_salario'], kde=True, ax=axes[1], color='green')
axes[1].set_title(f'Transformado con Log (Skew: {skewness_log:.2f})')

plt.tight_layout()
plt.show()

```

**Análisis Final:**
Notarás que el histograma transformado tiene una forma mucho más similar a la "campana de Gauss" (distribución normal). Los algoritmos como Regresión Lineal o Redes Neuronales "aprenden" mucho mejor de la variable `log_salario` que de la variable original con outliers.