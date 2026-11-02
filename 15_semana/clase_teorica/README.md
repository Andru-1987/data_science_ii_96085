# Modelos de Ensamble

Los métodos de ensamble combinan varios modelos para obtener un predictor más estable y preciso que sus componentes individuales. Su lógica central es gestionar el compromiso entre sesgo, varianza y error irreducible: **Bagging** prioriza reducir varianza, **Boosting** reduce principalmente el sesgo, y **Stacking** aprende a combinar modelos diversos.

---

## 1. Introducción a los Métodos de Ensamble

Un **ensamble** es una arquitectura de decisión distribuida: varios estimadores producen señales parciales y un mecanismo de agregación las convierte en una predicción final. El objetivo es lograr que los errores no sistemáticos de los modelos base se compensen entre sí.

El error esperado de un predictor se conceptualiza como:

$$\text{Error esperado} \approx \text{Sesgo}^2 + \text{Varianza} + \text{Error irreducible}$$

* **Sesgo alto:** El modelo es demasiado simple y no captura la estructura real (*underfitting*).
* **Varianza alta:** El modelo cambia demasiado ante pequeñas variaciones del conjunto de entrenamiento (*overfitting*).
* **Error irreducible:** Ruido inherente, información faltante o aleatoriedad real del fenómeno.

La efectividad de un ensamble depende de dos condiciones: que los modelos base sean **precisos** (mejores que el azar) y **diversos** (cometan errores distintos).

```text
Datos de entrada
      │
      ├── Modelo base 1 ──┐
      ├── Modelo base 2 ──┼── Agregador ── Predicción final
      ├── Modelo base 3 ──┘
      │
Diversidad + agregación = menor error generalizable

```

### Estrategias y Trade-offs Generales

| Estrategia | Mecanismo central | Ventajas | Desventajas | Caso de uso |
| --- | --- | --- | --- | --- |
| **Modelo único** | Aprende directamente de los datos. | Simple, interpretable y barato. | Puede tener sesgo o varianza alta. | Prototipos, reglas simples, datos escasos. |
| **Bagging** | Modelos paralelos sobre muestras bootstrap; promedio/votación. | Reduce varianza, estabiliza modelos. | No reduce mucho el sesgo; mayor costo. | Árboles profundos, datos ruidosos. |
| **Boosting** | Modelos secuenciales que corrigen errores previos. | Alto poder predictivo; reduce sesgo. | Sensible a ruido, outliers y sobreajuste. | Tabular estructurado, competencias, ranking. |
| **Stacking** | Modelos heterogéneos + meta-modelo. | Aprovecha fortalezas complementarias. | Complejidad alta; riesgo de *data leakage*. | Sistemas con datos suficientes y señales diversas. |
| **Mezcla ponderada** | Combina predicciones con pesos fijos o aprendidos. | Simple y controlable. | Menos expresivo que stacking. | Ensamble rápido de modelos ya validados. |

> **Buenas prácticas de la industria:** Comienza siempre con un *baseline* simple y medible. Introduce ensambles solo si la ganancia justifica la complejidad operativa. Nunca optimices hiperparámetros usando el set de test.

---

## 2. Bagging (Bootstrap Aggregating)

**Bagging** entrena $M$ modelos independientes sobre $M$ muestras generadas por *bootstrapping* (extrayendo observaciones con reemplazo desde el conjunto original).

En clasificación se usa votación mayoritaria, y en regresión, el promedio:


$$\hat{y} = \frac{1}{M}\sum_{m=1}^{M} \hat{y}_m$$

```text
Dataset original
   ├── Bootstrap sample 1 ── Modelo 1 ──┐
   ├── Bootstrap sample 2 ── Modelo 2 ──┼── Agregación ── Predicción
   └── Bootstrap sample 3 ── Modelo 3 ──┘

```

**Random Forest** extiende Bagging añadiendo una selección aleatoria de variables en cada división del árbol, reduciendo la correlación entre ellos.

### Variantes de Bagging

| Diseño | Ventajas | Desventajas | Caso de uso |
| --- | --- | --- | --- |
| **Random Forest** | Paralelizable, robusto, excelente desempeño base. | No extrapola fuera del rango observado. | Clasificación/regresión generalista. |
| **Extra Trees** | Más aleatoriedad, entrenamiento más rápido. | Puede perder algo de precisión. | Prototipado rápido o datasets enormes. |
| **Bagging lineal** | Mayor estabilidad. | Ganancia limitada (los modelos lineales ya son estables). | Útil como experimento. |

> **Tip Práctico:** Usa árboles poco podados como base; su alta varianza es justamente lo que Bagging mitiga. Random Forest es un excelente *baseline*, pero ten cuidado con series temporales donde la extrapolación es necesaria.

---

## 3. Boosting

**Boosting** construye un modelo aditivo de forma secuencial. Cada modelo es un *weak learner* que se agrega para corregir los errores que el ensamble acumulado aún no captura.

$$F_M(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$

```text
Modelo 1 -> Genera errores/residuos
      │
Modelo 2 -> Aprende de esos errores
      │
Modelo 3 -> Aprende del error restante
      │
Suma ponderada = Predicción final

```

### Principales Algoritmos de Boosting

#### AdaBoost (Adaptive Boosting)

Comienza con pesos uniformes. Tras entrenar un modelo, aumenta el peso de las instancias mal clasificadas, forzando al siguiente modelo a prestar atención a los casos difíciles.

* **Pro:** Conceptualmente simple y eficaz.
* **Contra:** Muy sensible a ruido y *outliers*.
* **Uso:** Clasificación binaria con datos limpios y clasificadores muy débiles (ej. *stumps*).

#### Gradient Boosting

Generaliza Boosting mediante optimización por descenso de gradiente. Cada nuevo árbol aprende a predecir el pseudo-residuo (el gradiente negativo de la función de pérdida) del ensamble anterior.


$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

* **Pro:** Flexible (admite distintas funciones de pérdida).
* **Uso:** Opción predeterminada para regresión y clasificación en datos estructurados.

#### XGBoost (Extreme Gradient Boosting)

Añade regularización explícita a la función objetivo (penalizando la complejidad del árbol) e incorpora optimizaciones de hardware (paralelismo).

* **Pro:** Control de sobreajuste excepcional mediante regularización L1/L2.
* **Uso:** Rendimiento de estado del arte para datos tabulares medianos/grandes. Requiere validar bien los hiperparámetros.

#### LightGBM

Optimiza la velocidad usando un crecimiento de árbol *leaf-wise* (por hoja de mayor ganancia) y discretizando continuas mediante *histogram binning*.

* **Pro:** Extremadamente rápido y eficiente en memoria.
* **Contra:** Su crecimiento *leaf-wise* puede sobreajustar en datasets pequeños.
* **Uso:** Grandes volúmenes de datos donde la latencia de entrenamiento es crítica.

> **Tip Práctico general para Boosting:** Optimiza primero la capacidad del árbol y la tasa de aprendizaje ($\eta$). Una tasa de aprendizaje pequeña requiere más árboles, pero generaliza mejor. Siempre utiliza *early stopping* con un conjunto de validación separado.

---

## 4. Stacking

**Stacking** entrena múltiples modelos base heterogéneos y utiliza un **meta-modelo** para combinar sus predicciones. El meta-modelo aprende en qué escenarios confiar más en cada modelo base.

```text
                 ┌── Regresión Logística ──┐
Datos de entrada ├── Máquinas de Soporte ──┼── Predicciones ── Meta-modelo ── Final
                 └── Random Forest ────────┘

```

Para evitar fuga de datos (*data leakage*), el meta-modelo debe entrenarse estrictamente con predicciones *out-of-fold* (generadas sobre particiones que los modelos base no usaron para entrenarse).

### Diseños de Stacking

| Diseño | Mecanismo | Pros / Contras |
| --- | --- | --- |
| **Stacking con CV** | Usa validación cruzada para predicciones base. | Muy robusto frente a fugas, pero computacionalmente costoso. |
| **Blending** | Usa un único *holdout* para entrenar el meta-modelo. | Más simple y rápido, pero desperdicia datos de entrenamiento. |
| **Votación suave** | Promedia las probabilidades emitidas. | Expresivo, pero exige que los modelos estén bien calibrados. |

> **Tip Práctico:** Mantén el meta-modelo simple (ej. una regresión lineal o logística). Combina familias de algoritmos distintas para maximizar la diversidad.

---

## 5. Regularización

La regularización introduce restricciones matemáticas al modelo para reducir la varianza, aceptando un ligero aumento en el sesgo con el fin de mejorar la generalización fuera de muestra.

$$\text{Objetivo} = \text{Pérdida de ajuste} + \lambda \cdot \text{Penalización de complejidad}$$

| Tipo | Mecanismo y Efecto | Caso de uso ideal |
| --- | --- | --- |
| **Lasso ($L_1$)** | Penaliza el valor absoluto de los coeficientes ($\vert{}\beta\vert{}$). Puede llevarlos exactamente a cero. | Selección automática de variables en datasets de alta dimensionalidad. |
| **Ridge ($L_2$)** | Penaliza los coeficientes al cuadrado ($\beta^2$). Los encoge pero no los anula. | Predictores muy correlacionados (estabiliza la multicolinealidad). |
| **Elastic Net** | Combina $L_1$ y $L_2$. | Requiere selección de variables pero se necesita estabilidad entre correlaciones. |

> **Tip Práctico:** Es obligatorio estandarizar/escalar las variables numéricas antes de aplicar regularización geométrica (como $L_1$ o $L_2$).

---

## 6. Glosario y Conceptos Clave

* **Parámetro:** Valor aprendido internamente por el modelo durante el entrenamiento (ej. los pesos de una regresión).
* **Hiperparámetro:** Configuración externa definida *antes* del entrenamiento para controlar la capacidad o regularización.
* **Weak Learner:** Un modelo simple con un rendimiento apenas superior al azar, utilizado como bloque de construcción (ej. *stumps* en Boosting).
* **Bootstrap:** Técnica estadistica de muestreo aleatorio con reemplazo.
* **Out-of-fold (OOF):** Predicciones realizadas sobre segmentos de datos que no fueron vistos durante el entrenamiento de ese modelo específico.
* **Early Stopping:** Técnica para detener el entrenamiento iterativo tan pronto como la métrica en el conjunto de validación deje de mejorar.
