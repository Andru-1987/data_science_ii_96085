### 1. Marco Metodológico: CRISP-DM

El ciclo de vida de un proyecto de datos debe estar guiado por el negocio antes que por la tecnología. CRISP-DM proporciona esta estructura a través de los siguientes pilares:

- **Alineación comercial:** Los objetivos técnicos deben traducirse de las metas del negocio. El éxito se mide por el impacto económico, no solo por la precisión matemática.
- **Distribución del esfuerzo:** Entre el 50% y el 70% del tiempo de un proyecto se consume en la preparación de los datos, lo cual es crítico para la planificación realista de recursos.
- **Iteración y control:** Es un proceso flexible que requiere validación constante. Finaliza con la puesta en producción y exige planes de mantenimiento para monitorear la expiración del modelo a lo largo del tiempo.

---

### 2. Preprocesamiento e Ingeniería de Características

Antes de que un algoritmo pueda aprender, los datos deben ser transformados a un formato numérico comprensible, enriqueciendo su valor predictivo.

#### 2.1 Codificación de Variables Categóricas

La elección de la codificación depende estrictamente de la naturaleza de la variable (nominal u ordinal) y del algoritmo a utilizar.

| Técnica              | Tipo de Variable                 | Ventajas                                                        | Desventajas / Riesgos                                                               | Referencias                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------- | -------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **One-Hot Encoding** | Nominales (Sin orden intrínseco) | No asume jerarquías falsas. Compatible con cualquier algoritmo. | Maldición de la dimensionalidad. Multicolinealidad (requiere `drop='first'`).       | [github](https://github.com/ricmed/artigos-sobre-ciencia-de-dados/blob/main/0004-label_encodign_one_hot_encoding.md?utm_source=gemini), [datacamp](https://www.datacamp.com/es/tutorial/principal-component-analysis-in-python?utm_source=gemini), [telefonicatech](https://telefonicatech.com/blog/python-para-todos-tutorial-de-pca-en-5?utm_source=gemini), [educative](https://www.educative.io/blog/one-hot-encoding?utm_source=gemini) |
| **Label Encoding**   | Ordinales / Modelos de Árboles   | Baja dimensionalidad (1 sola columna). Rápido.                  | Asigna ordenamiento arbitrario (alfabético) que puede confundir a modelos lineales. | [educative](https://www.educative.io/blog/one-hot-encoding?utm_source=gemini)                                                                                                                                                                                                                                                                                                                                                                |
| **Ordinal Encoding** | Ordinales con jerarquía clara    | Permite definir el orden correcto explícitamente.               | Incompatible con variables estrictamente nominales.                                 | [educative](https://www.educative.io/blog/one-hot-encoding?utm_source=gemini)                                                                                                                                                                                                                                                                                                                                                                |

#### 2.2 Tratamiento de Características (Features)

El refinamiento de las variables de entrada se divide en crear nueva información a partir de la lógica de negocio, o extraer patrones matemáticos subyacentes.

| Enfoque                | Definición                                                                                    | Técnicas Principales                                                         | Casos de Uso                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Feature Creation**   | Creación de nuevas variables derivadas a partir de las existentes.                            | Interacciones, variables temporales, agregaciones, discretización (Binning). | Datos relacionales, series temporales, transacciones financieras. |
| **Feature Extraction** | Transformación matemática de datos brutos hacia un nuevo espacio dimensional más informativo. | PCA (Varianza), NMF (Datos no negativos), TF-IDF (Frecuencia en texto).      | Procesamiento de lenguaje natural, imágenes, alta cardinalidad.   |

---

### 3. SECCIÓN IMPORTANTE: Mejoras del Modelo de Machine Learning

Un modelo base (entrenado con parámetros por defecto) rara vez alcanza el máximo potencial explicativo de los datos. La aplicación de técnicas de optimización y validación cruzada marca la diferencia entre un modelo de prueba y un sistema listo para producción.

#### 3.1 Impacto de la Optimización: Modelo Base vs. Modelo Optimizado

| Aspecto de Evaluación      | Modelo Base (Configuración por Defecto)                                       | Modelo Optimizado (Con Ajuste de Hiperparámetros y CV)                                                        |
| -------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Rendimiento Predictivo** | Subóptimo. Alto riesgo de estancarse en mínimos locales o memorizar el ruido. | Maximizado. El modelo se adapta a la complejidad real del conjunto de datos.                                  |
| **Método de Validación**   | División simple (Train/Test). Alta varianza en las métricas resultantes.      | Validación Cruzada (CV). Estimación robusta usando todos los datos para validación [suspicious link removed]. |
| **Control de Complejidad** | Estático. Vulnerable a sufrir Underfitting u Overfitting sin control.         | Dinámico. Los parámetros de regularización y profundidad se calibran matemáticamente.                         |
| **Costo Computacional**    | Muy bajo (un único ciclo de entrenamiento).                                   | Moderado a Alto (requiere múltiples iteraciones y paralelización).                                            |

#### 3.2 Estrategias de Búsqueda de Hiperparámetros

Para transicionar de un modelo base a uno optimizado, Scikit-Learn y herramientas afines ofrecen distintas metodologías de búsqueda.

| Técnica                | Mecanismo de Búsqueda                                                      | Velocidad                              | Calidad de la Solución                         | Ideal para                                                           | Referencias                                                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GridSearchCV**       | Exhaustiva (evalúa todo el grid).                                          | Lenta.                                 | Óptima dentro del espacio definido.            | Espacios de búsqueda pequeños (< 100 combinaciones).                 | [suspicious link removed]                                                                                                                                  |
| **RandomizedSearchCV** | Muestreo aleatorio mediante distribuciones estadísticas.                   | Rápida.                                | Buena (no garantiza el óptimo absoluto).       | Espacios grandes y primeros experimentos de modelado.                | [suspicious link removed], [jakevdp.github](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html?utm_source=gemini) |
| **HalvingSearchCV**    | Adaptativa (Successive Halving). Elimina temprano a los peores candidatos. | Muy rápida.                            | Buena.                                         | Entornos con muchísimos candidatos y recursos limitados.             | [suspicious link removed]                                                                                                                                  |
| **BayesSearchCV**      | Modelo probabilístico (Gaussian Process) para muestreo inteligente.        | Moderada (overhead propio del modelo). | Excelente. Balancea exploración y explotación. | Problemas costosos computacionalmente donde cada evaluación es cara. | [jakevdp.github](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html?utm_source=gemini)                            |

#### 3.3 Mitigación de Errores de Ajuste (Underfitting y Overfitting)

La optimización busca encontrar el punto de equilibrio entre un modelo demasiado simple y uno excesivamente memorizado.

| Problema         | Síntoma Principal                                 | Estrategias de Solución                                                                                 | Referencias Implicadas    |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------- |
| **Underfitting** | Mal rendimiento en entrenamiento y en prueba.     | Aumentar complejidad (más árboles, profundidad), añadir características, reducir regularización.        | N/A                       |
| **Overfitting**  | Excelente en entrenamiento, deficiente en prueba. | Validación cruzada, aplicar regularización (L1/L2), Feature Selection, Early Stopping, Dropout/Bagging. | [suspicious link removed] |

**Flujo de trabajo recomendado en producción:**
Comenzar estableciendo un Baseline (modelo base), explorar rápidamente con RandomizedSearchCV, refinar con HalvingRandomSearchCV, y hacer una validación final estricta sobre un set de datos aislado (test set) para evitar filtración de información (data leakage) [suspicious link removed]. Integrar siempre el preprocesamiento dentro de un `Pipeline` de Scikit-Learn garantiza la robustez del flujo.
