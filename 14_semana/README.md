 # Validación y Problemas en Machine Learning


El objetivo central de la evaluación de modelos no es maximizar el desempeño sobre datos conocidos, sino estimar y mejorar la **capacidad de generalización** sobre datos no vistos. El equilibrio entre sesgo y varianza, junto con un esquema de validación adecuado, determina si un modelo está subajustado, sobreajustado o correctamente calibrado. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html) 

## 1. Dialéctica de Generalización: Bias, Varianza y Complejidad

El objetivo prioritario de un modelo no es memorizar los datos de entrenamiento (*fit*), sino estimar el comportamiento con observaciones del mundo real no vistas.
 
 ![Bias vs Varianza](./.img/bias_vs_varianza.jpeg)

### Descomposición Teórica del Error

El error esperado en modelos supervisados se divide en tres componentes:

$$\text{Error Total} = \text{Bias}^2 + \text{Varianza} + \sigma^2$$

1. **Sesgo ($\text{Bias}^2$):** Error sistemático. Ocurre cuando el modelo impone supuestos demasiado restrictivos sobre la realidad (ej. ajustar un modelo lineal a una relación cuadrática). Produce **underfitting**.
2. **Varianza:** Inestabilidad o fotosensibilidad del modelo. Ocurre cuando la arquitectura absorbe demasiado el ruido específico o correlaciones espurias del dataset de entrenamiento. Produce **overfitting**.
3. **Error Irreducible ($\sigma^2$):** Ruido estocástico(aleatorio o probabilístico) del proceso generador de datos (o falta de variables observadas). Es el límite teórico máximo de precisión (*Error de Bayes*).

### Diagnóstico Operativo

| Estado | Error Train | Error Test/Val | Brecha Train-Test | Causa Raíz | Solución Directa |
| --- | --- | --- | --- | --- | --- |
| **Underfitting** | Alto | Alto | Muy baja | Falta de capacidad representacional | Añadir variables, transformaciones complejas, reducir regularización |
| **Buen Ajuste** | Bajo / Moderado | Bajo / Moderado | Pequeña | Equilibrio óptimo entre señal y ruido | Mantener arquitectura y esquema de validación |
| **Overfitting** | Muy bajo | Alto | Grande | Exceso de complejidad; memorización de ruido | Regularización ($L_1/L_2$), ensamble (Bagging), simplificación, más datos |

---

## 2. Estrategias y Esquemas de Validación

Para garantizar que las métricas obtenidas sean representativas del desempeño real, se implementan diferentes esquemas de partición según la naturaleza y volumen de los datos.

```
                                Dataset Completo
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Conjunto de Desarrollo (80%)                    Test Set Final (20%)
   (Cross-Validation / Hyperparameters)            (Evaluación final imparcial)

```

### Tipos de Validación

* **Validación Simple (Hold-out Train/Test Split):**
* *Mecanismo:* Una única división (ej. 80/20).
* *Cuándo usarlo:* Datasets masivos ($>10^6$ filas) o modelos extremadamente costosos de entrenar.
* *Riesgo:* Inestable y sensible a la aleatoriedad del corte.


* **K-Fold Cross-Validation:**
* *Mecanismo:* Se divide el conjunto en $K$ bloques. Se itera $K$ veces entrenando con $K-1$ y validando con el bloque restante.
* *Cuándo usarlo:* Estándar en la industria para evaluar estabilidad. Reportar siempre **Media $\pm$ Desviación Estándar** para cuantificar la dispersión entre *folds*.


* **Stratified K-Fold:**
* *Mecanismo:* Preserva la distribución porcentual exacta de las clases en cada uno de los $K$ *folds*.
* *Cuándo usarlo:* **Imprescindible** en clasificación con clases desbalanceadas (evita *folds* sin presencia de la clase minoritaria).


* **Leave-One-Out (LOOCV):**
* *Mecanismo:* Caso límite donde $K = N$ (se entrena $N$ veces omitiendo 1 sola observación por iteración).
* *Cuándo usarlo:* Exclusivamente en conjuntos de datos muy pequeños ($N < 100$) donde el costo computacional no sea una barrera.



---

## 3. Optimización Controlada de Hiperparámetros

Los hiperparámetros son decisiones externas al entrenamiento (ej. profundidad de un árbol, término de regularización $\lambda$) que configuran el nivel de complejidad del modelo.

```
                      Espacio de Búsqueda de Hiperparámetros
        ┌─────────────────────────────────────────────────────────────┐
        │  Profundidad, Regularización, Learning Rate, Min Samples    │
        │                                                             │
        │      ●     ●       ●                                        │
        │   ●        ●           ●        ●           ●               │
        │        ●           ●        ●                               │
        └─────────────────────────────────────────────────────────────┘
                                       ▲
                 Random Search: muestrea combinaciones óptimas
                      con un presupuesto fijo de tiempo/recursos.

```

### Estrategias de Búsqueda

1. **Grid Search:** Evalúa de forma exhaustiva el producto cartesiano de una grilla de parámetros. *Riesgo:* Explosión combinatoria/costo exponencial.
2. **Random Search:** Muestrea candidatos al azar desde distribuciones de probabilidad predefinidas. *Ventaja:* Mayor cobertura y flexibilidad en el espacio de búsqueda bajo un presupuesto de iteraciones fijo.
3. **Búsqueda Bayesiana:** Construye un modelo probabilístico del rendimiento para proponer la siguiente mejor combinación.

### El Protocolo de Evaluación Sin Sesgo

Para evitar la **fuga de información (Data Leakage)** al optimizar hiperparámetros:

1. Aislar de forma estricta un **Test Set Final** al inicio del proyecto.
2. Realizar el procesamiento, escalado/imputación de datos y la búsqueda de hiperparámetros **únicamente** dentro del loop de entrenamiento de cada *fold* del conjunto de desarrollo.
3. Reentrenar el modelo final con la mejor combinación de hiperparámetros sobre la totalidad del conjunto de desarrollo.
4. Evaluar **una única vez** sobre el Test Set Final para obtener la métrica definitiva de generalización.

---

## 4. Anti-Patrones Críticos en la Práctica Profesional

Evitar estos errores comunes es fundamental para asegurar la interpretabilidad y el éxito operativo de los modelos:

```txt
                            FLUJO CORRECTO SIN FUGA
                            
  Datos Brutos ──► Split Train / Test
                          │
                          ├──► Train Fold ──► Fit Scaler/Imputer ──► Transform
                          │
                          └──► Val Fold   ──► Use Fitted Parameters ──► Transform

```

* **Fuga de Información por Preprocesamiento (Data Leakage):**
* *Error:* Aplicar escaladores (`StandardScaler`), imputadores o selección de variables sobre todo el dataset antes de dividirlo o realizar Cross-Validation.
* *Corrección:* Ajustar (`.fit()`) cualquier transformación exclusivamente sobre el subconjunto de entrenamiento de cada *fold*.


* **Ajuste de Hiperparámetros sobre el Test Set:**
* *Error:* Tomar decisiones de arquitectura o hiperparámetros basándose en el resultado del Test Set.
* *Corrección:* El Test Set es un **contrato de auditoría final**; sólo se consulta cuando el modelo está completamente definido.


* **Uso de K-Fold Aleatorio en Datos Temporales:**
* *Error:* Usar división aleatoria cuando los datos dependen del tiempo (Series Temporales).
* *Corrección:* Usar validación temporal progresiva (*Time Series Split* / *Rolling Windows*) para respetar la causalidad.


* **Optimización de Métricas Incorrectas:**
* *Error:* Usar *Accuracy* como métrica objetivo en datasets severamente desbalanceados.
* *Corrección:* Alinear el modelo al dominio del problema mediante *Precision*, *Recall*, *PR-AUC* o funciones de costo personalizadas.
