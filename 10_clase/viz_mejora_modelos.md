1. **Gráfico de Barras Comparativo de Métricas ($R^2$ y RMSE):** Permite ver cuantitativamente cuánto mejora el modelo. Plotly nos permite inyectar los hiperparámetros en el `hover` (la ventana emergente al pasar el cursor) para que los alumnos vean exactamente qué configuración logró ese resultado.
2. **Gráfico de Dispersión (Actual vs. Predicho):** Muestra cualitativamente la mejora. Entre más cerca estén los puntos de una línea diagonal perfecta, mejor es el modelo.

### Código de Entrenamiento y Visualización Interactiva

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# 1. Carga de Datos y Preparación
# Asumiendo que el archivo se llama 'advertising.csv'
df = pd.read_csv('advertising.csv', index_col=0)

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Separamos en conjunto de entrenamiento y prueba para una evaluación justa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista para almacenar los resultados y graficarlos luego
resultados = []
predicciones = {'Real': y_test.values}

# ==========================================
# MODELO 1: Random Forest (Base - Sin tunear)
# ==========================================
rf_base = RandomForestRegressor(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

resultados.append({
    'Modelo': '1. Random Forest (Base)',
    'R2_Score': r2_score(y_test, y_pred_base),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_base)),
    'Hiperparametros': 'Valores por defecto de Scikit-Learn'
})
predicciones['RF_Base'] = y_pred_base

# ==========================================
# MODELO 2: Random Forest (Optimizado)
# ==========================================
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
y_pred_rf_opt = grid_rf.predict(X_test)

resultados.append({
    'Modelo': '2. Random Forest (Optimizado)',
    'R2_Score': r2_score(y_test, y_pred_rf_opt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf_opt)),
    'Hiperparametros': str(grid_rf.best_params_)
})
predicciones['RF_Opt'] = y_pred_rf_opt

# ==========================================
# MODELO 3: Gradient Boosting (Optimizado)
# ==========================================
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5]
}
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, scoring='r2', n_jobs=-1)
grid_gb.fit(X_train, y_train)
y_pred_gb_opt = grid_gb.predict(X_test)

resultados.append({
    'Modelo': '3. Gradient Boosting (Opt)',
    'R2_Score': r2_score(y_test, y_pred_gb_opt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb_opt)),
    'Hiperparametros': str(grid_gb.best_params_)
})
predicciones['GB_Opt'] = y_pred_gb_opt

# ==========================================
# MODELO 4: Support Vector Regression (SVR - Optimizado)
# ==========================================
param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
grid_svr.fit(X_train, y_train)
y_pred_svr_opt = grid_svr.predict(X_test)

resultados.append({
    'Modelo': '4. SVR (Optimizado)',
    'R2_Score': r2_score(y_test, y_pred_svr_opt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr_opt)),
    'Hiperparametros': str(grid_svr.best_params_)
})
predicciones['SVR_Opt'] = y_pred_svr_opt

# ==========================================
# VISUALIZACIÓN 1: Gráfico de Barras Interactivo
# ==========================================
df_resultados = pd.DataFrame(resultados)

# Gráfico para R2 Score (Más cercano a 1.0 es mejor)
fig_r2 = px.bar(
    df_resultados,
    x='Modelo',
    y='R2_Score',
    color='Modelo',
    text='R2_Score',
    hover_data={'Hiperparametros': True, 'Modelo': False}, # Muestra los hiperparámetros al pasar el mouse
    title="Comparación de Rendimiento Predictivo (R² Score) - Más alto es mejor",
    labels={'R2_Score': 'Coeficiente de Determinación (R²)'}
)
fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_r2.update_layout(yaxis_range=[0, 1.1])
fig_r2.show()

# ==========================================
# VISUALIZACIÓN 2: Gráfico de Dispersión (Real vs Predicho)
# ==========================================
# Seleccionamos el modelo Base y el mejor modelo (supongamos GB_Opt para el ejemplo)
fig_scatter = go.Figure()

# Línea ideal (Predicción perfecta)
rango_min = min(y_test) - 2
rango_max = max(y_test) + 2
fig_scatter.add_trace(go.Scatter(
    x=[rango_min, rango_max], y=[rango_min, rango_max],
    mode='lines',
    name='Predicción Perfecta',
    line=dict(color='black', dash='dash')
))

# Puntos del Modelo Base
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_pred_base,
    mode='markers',
    name='RF Base',
    marker=dict(color='red', size=8, opacity=0.6),
    hovertemplate='Real: %{x}<br>Predicho: %{y}'
))

# Puntos del Mejor Modelo (Ej: Gradient Boosting Optimizado)
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_pred_gb_opt,
    mode='markers',
    name='GB Optimizado',
    marker=dict(color='green', size=8, opacity=0.8),
    hovertemplate='Real: %{x}<br>Predicho: %{y}'
))

fig_scatter.update_layout(
    title="Análisis de Residuales: Ventas Reales vs. Ventas Predichas",
    xaxis_title="Ventas Reales (Ground Truth)",
    yaxis_title="Ventas Predichas por el Modelo",
    hovermode='closest'
)
fig_scatter.show()

```

- **Lectura del Gráfico 1 (Barras):** El objetivo es que la audiencia note el "delta" o salto de rendimiento entre la primera barra (modelo sin intervención humana) y el resto de las barras.
- **Selección de algoritmos:** Explicar por qué se agregaron _Gradient Boosting_ y _SVR_. El primero es un ensamble secuencial que suele dominar en datos tabulares, mientras que el SVR aporta una perspectiva geométrica diferente para la regresión.
- **Lectura del Gráfico 2 (Dispersión):** La línea punteada negra representa un modelo infalible ($y=x$). En la clase, puedes mostrar de forma interactiva cómo los puntos verdes (modelo optimizado) tienden a "abrazar" o agruparse mucho más cerca de la línea punteada en comparación con los puntos rojos (modelo base), los cuales presentan una dispersión más errática frente a valores atípicos.
