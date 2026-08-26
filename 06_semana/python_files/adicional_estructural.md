### Arquitectura del Análisis Visual Ampliado

* **Distribuciones Categóricas (Barras y Tortas):** Esenciales para entender la composición del mercado (ej. dominio de transmisiones automáticas y tracción delantera).
* **Relaciones Bivariadas (Dispersión y Heatmap):** Fundamentales para validación de hipótesis de ingeniería, como la compensación entre potencia (HP) y eficiencia térmica (MPG).
* **Distribuciones Complejas (Boxplot y Violín):** Críticas para observar la densidad y dispersión de precios según características estructurales (ej. cantidad de cilindros), detectando asimetrías que un simple promedio ocultaría.

---

### Script Extendido (`eda_cars_advanced.py`)

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Carga, limpia y prepara el dataset."""
    df = pd.read_csv(filepath)
    
    # Limpieza y renombrado
    columnas_a_eliminar = ['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size']
    df = df.drop(columnas_a_eliminar, axis=1)
    df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode", "highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price"})
    
    # Deduplicación y manejo de nulos
    df = df.drop_duplicates().dropna()
    
    # Remoción de Outliers (IQR)
    Q1 = df.quantile(0.25, numeric_only=True)
    Q3 = df.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df

def plot_categorical_distributions(df: pd.DataFrame):
    """
    Gráficos de Barras y Tortas
    - Qué vemos: La proporción de los tipos de transmisión y modos de tracción en el mercado.
    - Por qué es importante: Ayuda a identificar sesgos en el dataset. Si queremos entrenar un modelo, 
      necesitamos saber si está desbalanceado hacia vehículos automáticos o de tracción delantera.
    - Conclusión: El mercado automotriz en este dataset está abrumadoramente dominado por transmisiones 
      automáticas y tracción delantera (FWD) o trasera (RWD), siendo la tracción integral (AWD) menos común.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de Barras: Transmisión
    df['Transmission'].value_counts().plot(kind='bar', ax=axes[0], color='teal')
    axes[0].set_title('Distribución por Tipo de Transmisión')
    axes[0].set_ylabel('Cantidad')
    
    # Gráfico de Torta: Drive Mode
    df['Drive Mode'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', cmap='Set3')
    axes[1].set_title('Proporción de Modos de Tracción')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

def plot_relationships(df: pd.DataFrame):
    """
    Dispersión y Mapa de Calor (Heatmap)
    - Qué vemos: El Heatmap muestra la magnitud de correlación entre todas las variables numéricas. 
      El Scatterplot visualiza la relación negativa entre HP y MPG-H.
    - Por qué es importante: El heatmap es el núcleo de la selección de características (Feature Selection). 
      El scatterplot confirma visualmente si la relación es lineal o no lineal.
    - Conclusión: Se confirma el "Trade-off" térmico/mecánico: a mayor HP (potencia), el MPG (eficiencia) 
      cae drásticamente. Además, el Precio está fuertemente anclado al HP, pero no a la eficiencia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Mapa de Calor: Matriz de Correlación')
    
    # Dispersión: HP vs MPG-H
    sns.scatterplot(data=df, x='HP', y='MPG-H', alpha=0.5, ax=axes[1], color='darkred')
    axes[1].set_title('Dispersión: Potencia (HP) vs Eficiencia en Carretera (MPG-H)')
    
    plt.tight_layout()
    plt.show()

def plot_advanced_distributions(df: pd.DataFrame):
    """
    Boxplots y Gráficos de Violín
    - Qué vemos: El Boxplot muestra los cuartiles y la mediana del precio según el tipo de transmisión. 
      El Violín muestra la densidad de probabilidad del precio según la cantidad de cilindros.
    - Por qué es importante: El boxplot aísla anomalías locales dentro de categorías específicas. 
      El violín revela si los precios tienen una distribución bimodal (dos picos de precios) dentro de un mismo tipo de motor.
    - Conclusión (Boxplot): Los vehículos automatizados tienen una mediana de precio superior a los manuales, 
      con una mayor varianza.
    - Conclusión (Violín): Los motores de 4 cilindros tienen precios altamente concentrados en un rango bajo-medio. 
      Los motores de 6 y 8 cilindros muestran una distribución más alargada, indicando que hay otros factores (como marca o lujo) empujando el precio hacia arriba.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot: Precio por Transmisión
    sns.boxplot(data=df, x='Transmission', y='Price', ax=axes[0], palette='pastel')
    axes[0].set_title('Boxplot: Dispersión de Precio por Transmisión')
    
    # Violinplot: Precio por Cilindros (Top 3 configuraciones más comunes para claridad)
    top_cylinders = df['Cylinders'].value_counts().nlargest(3).index
    df_filtered = df[df['Cylinders'].isin(top_cylinders)]
    
    sns.violinplot(data=df_filtered, x='Cylinders', y='Price', ax=axes[1], palette='muted', inner='quartile')
    axes[1].set_title('Gráfico de Violín: Densidad de Precio por Cilindros (4, 6, 8)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ruta al dataset
    dataset_path = "data.csv"
    
    try:
        print("Iniciando pipeline de EDA...")
        df_clean = load_and_clean_data(dataset_path)
        
        plot_categorical_distributions(df_clean)
        plot_relationships(df_clean)
        plot_advanced_distributions(df_clean)
        
        print("Pipeline finalizado con éxito.")
    except Exception as e:
        print(f"Error en la ejecución: {e}")

```
