# Validación y Problemas en Machine Learning

El objetivo central de la evaluación de modelos no es maximizar el desempeño sobre datos conocidos, sino estimar y mejorar la **capacidad de generalización** sobre datos no vistos. El equilibrio entre sesgo y varianza, junto con un esquema de validación adecuado, determina si un modelo está subajustado, sobreajustado o correctamente calibrado. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

***

## 14.1 Bias y Varianza

### Fundamentos e internos

El **sesgo** (*bias*) mide el error sistemático que surge cuando la familia de modelos es demasiado simple para representar la relación real entre variables. Un modelo con alto sesgo impone supuestos restrictivos y produce predicciones consistentemente alejadas de la función generadora de los datos. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

La **varianza** mide cuánto cambia el modelo aprendido cuando se entrena con distintas muestras del mismo proceso generador. Un modelo de alta varianza es excesivamente sensible al conjunto específico de entrenamiento, capturando ruido, valores atípicos y correlaciones espurias. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

```text
Datos reales
     │
     ├── Señal estructural estable ──► debe ser aprendida
     └── Ruido aleatorio ────────────► no debe ser aprendida

Modelo simple
     ├── Alto bias
     ├── Baja varianza
     └── No captura suficiente señal → Underfitting

Modelo excesivamente flexible
     ├── Bajo bias
     ├── Alta varianza
     └── Aprende señal + ruido → Overfitting
```

| Dimensión | Bias alto | Varianza alta |
|---|---|---|
| Causa típica | Modelo demasiado simple o restricciones excesivas | Modelo demasiado complejo respecto a los datos |
| Comportamiento en entrenamiento | Error alto | Error muy bajo |
| Comportamiento en prueba | Error alto | Error alto |
| Diagnóstico | Underfitting | Overfitting |
| Predicciones | Sistemáticamente desviadas | Inestables ante cambios de muestra |
| Remedio principal | Aumentar capacidad representacional | Regularizar, simplificar o ampliar datos |

El error esperado de predicción puede descomponerse, en el contexto clásico de regresión cuadrática, como:

\[
\text{Error Total} =
\underbrace{\text{Bias}^2}_{\text{error sistemático}}
+
\underbrace{\text{Varianza}}_{\text{inestabilidad}}
+
\underbrace{\sigma^2}_{\text{error irreducible}}
\]

El término irreducible corresponde al ruido inherente del proceso generador, también llamado error de Bayes; no puede eliminarse mediante más datos, hiperparámetros ni mayor capacidad del modelo. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

### Trade-off Bias-Varianza

El compromiso fundamental establece que aumentar la flexibilidad del modelo suele reducir el sesgo pero incrementar la varianza; reducir la flexibilidad hace lo contrario. El objetivo no es minimizar aisladamente bias o varianza, sino minimizar su suma, que constituye el error reducible. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

```text
Error
  ▲
  │ Error total
  │      ╲________________
  │       ╲      _________╲──── Error irreducible
  │        ╲____╱
  │      ┌──┴──┐
  │ Bias²│     │ Varianza
  │  ╲   │     │   ╱
  │   ╲  │     │  ╱
  └──────┴─────┴──────────────► Complejidad del modelo
       Underfitting   Overfitting
              ▲
        Zona de mejor generalización
```

| Nivel de complejidad | Bias | Varianza | Riesgo |
|---|---|---|---|
| Muy bajo | Alto | Bajo | Underfitting |
| Intermedio | Moderado | Moderado | Mejor equilibrio |
| Muy alto | Bajo | Alto | Overfitting |

### Patrones de diseño y arquitectura

En sistemas de ML, el control de bias y varianza debe incorporarse como una decisión de arquitectura, no como un ajuste tardío.

- **Pipeline reproducible**: separar preparación de datos, entrenamiento, validación y evaluación final.
- **Control de capacidad**: definir límites explícitos de complejidad, como profundidad máxima, número de parámetros, número de reglas o tamaño de vecindario.
- **Regularización como política**: aplicar restricciones que penalicen modelos innecesariamente complejos.
- **Validación como contrato**: toda métrica reportada debe provenir de datos no utilizados para ajustar parámetros.
- **Observabilidad de generalización**: registrar brecha entre desempeño de entrenamiento y validación como señal de sobreajuste.

```text
Datos brutos
     │
     ▼
Preparación / transformación
     │
     ├──► Conjunto de entrenamiento ──► Ajuste de parámetros
     ├──► Conjunto de validación ─────► Selección de hiperparámetros
     └──► Conjunto de prueba ─────────► Estimación final imparcial
```

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Comparar siempre error de entrenamiento y error de validación; la brecha entre ambos es más informativa que cualquiera de los dos por separado.
- Ajustar hiperparámetros únicamente con datos de validación o validación cruzada.
- Reservar un conjunto de prueba final que no participe en ninguna decisión de diseño.
- Documentar la métrica, el esquema de partición, las semillas y la versión del dataset.
- Preferir modelos más simples cuando la mejora en validación sea marginal.

**Anti-patrones**

| Anti-patrón | Consecuencia | Corrección |
|---|---|---|
| Optimizar hiperparámetros sobre el test set | Contamina la estimación final de generalización | Usar validación interna y test final aislado |
| Interpretar alto score de entrenamiento como buen modelo | Puede ocultar overfitting severo | Comparar train vs. validación/test |
| Aumentar complejidad sin más datos ni regularización | Incrementa varianza | Controlar capacidad y validar |
| Aplicar transformaciones antes de dividir datos | Puede introducir fuga de información | Ajustar transformaciones solo con datos de entrenamiento |
| Elegir el modelo con mejor resultado de una sola partición | Resultado sensible al azar de la división | Usar validación cruzada y reportar dispersión |

***

## 14.2 Overfitting y Underfitting

### Fundamentos e internos

El **overfitting** ocurre cuando el modelo aprende patrones específicos del conjunto de entrenamiento que no se sostienen en la población. Se manifiesta como error bajo en entrenamiento y error alto en validación o prueba. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

El **underfitting** ocurre cuando el modelo carece de capacidad suficiente para capturar la estructura relevante. Se manifiesta como error alto tanto en entrenamiento como en prueba. [statisticallearning](https://statisticallearning.org/bias-variance-tradeoff.html)

```text
Underfitting
Train error: alto
Test error:  alto
Brecha:      baja
Diagnóstico: modelo incapaz de aprender la señal

Buen ajuste
Train error: bajo-moderado
Test error:  bajo-moderado
Brecha:      pequeña
Diagnóstico: generalización adecuada

Overfitting
Train error: muy bajo
Test error:  alto
Brecha:      grande
Diagnóstico: memorización del ruido
```

| Problema | Señal diagnóstica | Causa frecuente | Intervención |
|---|---|---|---|
| Overfitting | Error bajo en train, alto en test | Exceso de capacidad, pocos datos, ruido | Regularización, más datos, simplificación, validación cruzada |
| Underfitting | Error alto en train y test | Modelo restrictivo, features insuficientes | Mayor complejidad, mejores variables, transformaciones |
| Fuga de datos | Métricas excelentes pero mal desempeño operativo | Información del futuro o del target presente en features | Revisar pipeline y temporalidad |
| Dataset no representativo | Buen resultado en validación, mal resultado en producción | Sesgo de muestreo o deriva de distribución | Reentrenar con datos representativos y monitorear |

### Patrones de diseño y arquitectura

- **Presupuesto de complejidad**: establecer límites de capacidad antes del entrenamiento.
- **Selección basada en validación**: elegir el modelo por desempeño fuera de muestra, no por ajuste interno.
- **Regularización estructural**: incorporar penalizaciones o restricciones de complejidad al proceso de aprendizaje.
- **Aumento controlado de datos**: usar más observaciones o augmentación sólo cuando represente fielmente el dominio real.
- **Ensamble**: combinar múltiples modelos para reducir varianza mediante agregación o mejorar sesgo mediante aprendizaje secuencial.

| Técnica | Efecto predominante | Mecanismo conceptual |
|---|---|---|
| Regularización \(L_2\) | Reduce varianza | Penaliza magnitudes grandes de parámetros |
| Regularización \(L_1\) | Reduce varianza y puede seleccionar variables | Penaliza magnitudes y favorece soluciones dispersas |
| Más datos | Reduce varianza | Disminuye dependencia de una muestra particular |
| Modelo más simple | Aumenta bias, reduce varianza | Limita capacidad de memorización |
| Modelo más complejo | Reduce bias, aumenta varianza | Amplía el espacio de funciones representables |
| Bagging | Reduce varianza | Promedia modelos entrenados sobre submuestras |
| Boosting | Puede reducir bias | Construye modelos secuencialmente para corregir errores |

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Diagnosticar antes de intervenir: distinguir entre underfitting, overfitting, fuga de datos y mala calidad de etiquetas.
- Incrementar complejidad gradualmente y medir su efecto en validación.
- Aplicar regularización cuando el modelo tenga muchos parámetros o los datos sean escasos.
- Usar validación cruzada para estimar la estabilidad del desempeño.
- Tratar la brecha train-test como una señal operativa, no como un número aislado.

**Anti-patrones**

- Añadir features sin justificación causal o estadística: aumenta dimensionalidad y riesgo de sobreajuste.
- Eliminar features únicamente porque un modelo simple no las aprovecha: puede aumentar sesgo.
- Usar una sola partición para decidir arquitectura: puede conducir a una elección azarosa.
- Confundir sobreajuste con mala calidad de datos: ambos pueden producir mal desempeño en test, pero requieren soluciones distintas.

***

## 14.3 Técnicas de Validación

### Fundamentos e internos

La validación estima el comportamiento del modelo sobre datos no utilizados en el entrenamiento. Su propósito es reducir la dependencia del resultado respecto a una partición particular y obtener una estimación más estable de la generalización. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

#### Validación simple

La validación simple divide los datos en entrenamiento y prueba. Es computacionalmente barata, pero produce una única estimación que puede variar considerablemente según la partición elegida. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

```text
Dataset completo
┌───────────────────────────────┬───────────────┐
│        Entrenamiento          │     Prueba    │
└───────────────────────────────┴───────────────┘
```

| Aspecto | Validación simple |
|---|---|
| Costo computacional | Bajo |
| Estabilidad de estimación | Baja |
| Uso de datos | Una sola partición |
| Riesgo principal | Resultado dependiente del azar de la división |
| Uso recomendado | Prototipado, datasets grandes, entrenamientos costosos |

#### K-Fold Cross-Validation

En K-Fold, el dataset se divide en \(k\) particiones o *folds*. En cada iteración, \(k-1\) folds se usan para entrenamiento y el fold restante para validación; el resultado final se obtiene promediando las \(k\) evaluaciones. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

```text
Fold 1: [ TEST ][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][ TEST ][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][ TEST ][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][ TEST ][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][ TEST ]

Score final = media(score_1, ..., score_5)
Dispersión  = desviación estándar(score_1, ..., score_5)
```

| Aspecto | K-Fold |
|---|---|
| Costo computacional | \(k\) entrenamientos |
| Estabilidad | Alta respecto a una partición simple |
| Uso de datos | Cada observación se valida exactamente una vez |
| Salida recomendada | Media y desviación estándar |
| Riesgo | Fuga si el preprocesamiento no se ajusta dentro de cada fold |

#### Stratified K-Fold

Stratified K-Fold conserva aproximadamente la proporción de cada clase de la variable objetivo en cada fold. Es especialmente apropiado para clasificación con clases desbalanceadas, porque evita que algún fold contenga muy pocos ejemplos de la clase minoritaria. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

```text
Distribución original: 80% clase 0, 20% clase 1

Fold 1: 80% clase 0, 20% clase 1
Fold 2: 80% clase 0, 20% clase 1
Fold 3: 80% clase 0, 20% clase 1
Fold 4: 80% clase 0, 20% clase 1
Fold 5: 80% clase 0, 20% clase 1
```

| Aspecto | K-Fold estándar | Stratified K-Fold |
|---|---|---|
| Preserva proporción de clases | No garantizado | Sí, aproximadamente |
| Adecuado para clasificación desbalanceada | Riesgoso | Recomendado |
| Adecuado para regresión | Sí | No aplica directamente |
| Riesgo principal | Fold sin representación de minoritaria | Requiere suficiente tamaño por clase |

#### Leave-One-Out Cross-Validation

LOOCV es un caso extremo de validación cruzada donde \(k = N\): cada observación se usa una vez como conjunto de prueba y las \(N-1\) restantes como entrenamiento. Produce \(N\) modelos y \(N\) evaluaciones. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

```text
Iteración 1: [TEST: x1] [TRAIN: x2 ... xN]
Iteración 2: [TRAIN: x1] [TEST: x2] [TRAIN: x3 ... xN]
...
Iteración N: [TRAIN: x1 ... xN-1] [TEST: xN]
```

| Aspecto | LOOCV |
|---|---|
| Número de entrenamientos | \(N\) |
| Aprovechamiento de datos | Máximo por iteración |
| Precisión estadística | Alta, con baja dependencia de una partición |
| Costo computacional | Muy alto |
| Riesgo principal | Inviabilidad en datasets grandes o modelos costosos |
| Uso recomendado | Datasets pequeños y modelos de entrenamiento económico |

### Patrones de diseño y arquitectura

En arquitecturas de ML serias, la validación debe formar parte del ciclo de vida del modelo.

```text
Datos versionados
        │
        ▼
Esquema de validación
        │
        ├── Entrenamiento por fold
        ├── Evaluación por fold
        ├── Agregación de métricas
        │
        ▼
Selección de modelo / hiperparámetros
        │
        ▼
Conjunto de prueba final aislado
        │
        ▼
Registro de artefactos y decisión de despliegue
```

- **Validación anidada**: usar validación cruzada interna para seleccionar hiperparámetros y validación externa para estimar generalización.
- **Validación temporal**: en datos con dimensión temporal, entrenar con el pasado y validar con el futuro; K-Fold aleatorio puede violar la causalidad.
- **Validación por grupos**: cuando existen entidades repetidas, como usuarios o pacientes, todas las filas de una entidad deben permanecer en el mismo lado de la división.
- **Trazabilidad**: almacenar folds, semillas, métricas por fold y versión del modelo.

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Elegir \(k\) según el tamaño del dataset, el costo de entrenamiento y la varianza esperada de las métricas.
- Reportar media y desviación estándar en validación cruzada; la dispersión revela inestabilidad.
- Usar Stratified K-Fold en clasificación desbalanceada.
- Aplicar todo preprocesamiento dentro del esquema de validación para evitar fuga de información.
- Mantener un test final separado para la estimación definitiva.

**Anti-patrones**

| Anti-patrón | Riesgo | Mitigación |
|---|---|---|
| Escalar, imputar o seleccionar features antes de dividir | Fuga de información | Ajustar transformaciones sólo con datos de entrenamiento de cada fold |
| Usar K-Fold aleatorio en series temporales | El modelo puede “ver el futuro” | Usar partición temporal progresiva |
| Usar LOOCV en datasets grandes | Costo computacional prohibitivo | Usar K-Fold o validación simple |
| Elegir hiperparámetros con el test set | Sobreestimación de generalización | Reservar test final |
| Ignorar la desviación estándar entre folds | Selección de modelos frágiles | Comparar media y dispersión |
| Validar con grupos mezclados | Fuga por entidades repetidas | Validación agrupada por entidad |

***

## 14.4 Optimización de Hiperparámetros

### Fundamentos e internos

Los **hiperparámetros** son decisiones de configuración que no se aprenden directamente durante el entrenamiento, sino que se seleccionan antes o durante el proceso de modelado. Controlan la capacidad del modelo y, por ende, su posición en el espectro bias-varianza. [scikit-learn](https://scikit-learn.org/dev/modules/grid_search.html)

La búsqueda en grilla evalúa combinaciones predefinidas de forma exhaustiva; la búsqueda aleatoria muestrea un número limitado de configuraciones desde distribuciones definidas sobre el espacio de hiperparámetros. [scikit-learn](https://scikit-learn.org/dev/modules/grid_search.html)

| Estrategia | Mecanismo | Ventaja | Limitación |
|---|---|---|---|
| Grid Search | Evalúa todas las combinaciones especificadas | Cobertura completa del espacio definido | Costo exponencial con muchos parámetros |
| Random Search | Muestrea combinaciones aleatorias | Explora más diversidad con presupuesto fijo | No garantiza cubrir el mejor punto |
| Búsqueda bayesiana | Modela el desempeño y propone candidatos prometedores | Eficiente en espacios costosos | Mayor complejidad de implementación |
| Optimización evolutiva | Evoluciona poblaciones de configuraciones | Útil en espacios discretos y complejos | Requiere muchas evaluaciones |

```text
Espacio de hiperparámetros
┌────────────────────────────────────────────┐
│  profundidad, regularización, tamaño, etc. │
│                                            │
│      ●  ●     ●                            │
│   ●        ●        ●        ●             │
│        ●         ●      ●                  │
│             ●                              │
└────────────────────────────────────────────┘
        ▲
Búsqueda aleatoria: evalúa un subconjunto
representativo bajo un presupuesto fijo
```

### Trade-off clave

| Decisión | Beneficio | Costo |
|---|---|---|
| Más iteraciones de búsqueda | Mayor probabilidad de encontrar buena configuración | Más entrenamientos y tiempo |
| Espacio más amplio | Mayor exploración | Mayor riesgo de configuraciones irrelevantes |
| Validación con más folds | Estimación más estable | Mayor costo computacional |
| Métrica alineada al negocio | Optimiza el objetivo correcto | Puede requerir métricas personalizadas |
| Test final aislado | Estimación honesta | Reduce datos disponibles para entrenamiento/validación |

### Patrones de diseño y arquitectura

- **Búsqueda en dos etapas**: exploración amplia con búsqueda aleatoria y refinamiento local alrededor de las mejores regiones.
- **Optimización con validación cruzada**: cada candidato debe evaluarse mediante el mismo esquema de validación.
- **Selección por métrica de negocio**: Accuracy puede ser insuficiente en clases desbalanceadas; conviene considerar Precision, Recall, F1, AUC o una función de costo.
- **Registro de experimentos**: guardar configuración, folds, métricas, duración y artefacto del modelo.
- **Reentrenamiento controlado**: tras seleccionar hiperparámetros, reentrenar sobre el conjunto de entrenamiento completo según la política definida y evaluar una sola vez en test.

```text
Conjunto de entrenamiento
        │
        ├──► Validación cruzada interna
        │         │
        │         ├── Candidato 1 ──► score
        │         ├── Candidato 2 ──► score
        │         └── Candidato N ──► score
        │
        ▼
Mejor configuración
        │
        ▼
Conjunto de prueba final
        │
        ▼
Métrica final de generalización
```

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Definir primero la métrica de evaluación y después el espacio de búsqueda.
- Usar distribuciones en lugar de listas fijas cuando el parámetro sea continuo o tenga muchas opciones válidas.
- Limitar el presupuesto de búsqueda mediante un número fijo de iteraciones.
- Incluir restricciones de complejidad en el espacio de búsqueda para controlar varianza.
- Repetir la evaluación final sólo una vez sobre el test reservado.

**Anti-patrones**

| Anti-patrón | Consecuencia | Corrección |
|---|---|---|
| Buscar hiperparámetros sobre todo el dataset | Fuga y sobreestimación | Buscar sólo en train/validación |
| Optimizar Accuracy en clases desbalanceadas | Puede favorecer la clase mayoritaria | Optimizar métrica sensible al desbalance |
| Buscar sin presupuesto | Costo computacional incontrolable | Definir número de iteraciones o tiempo máximo |
| Elegir el mejor score sin considerar dispersión | Modelo frágil | Analizar media y desviación entre folds |
| Tratar hiperparámetros como parámetros del modelo | Confusión conceptual | Distinguir parámetros aprendidos de configuraciones externas |

***

## 14.5 Glosario Técnico

| Término | Definición operativa |
|---|---|
| **Bias** | Error sistemático por supuestos o capacidad representacional insuficiente |
| **Varianza** | Sensibilidad del modelo ante cambios en la muestra de entrenamiento |
| **Underfitting** | Alto error en entrenamiento y prueba por modelo demasiado simple |
| **Overfitting** | Bajo error en entrenamiento y alto error en prueba por memorización |
| **Error irreducible** | Ruido inherente del proceso generador; no eliminable por el modelo |
| **Validación simple** | Una división entrenamiento/prueba |
| **K-Fold** | Validación cruzada con \(k\) particiones rotativas |
| **Stratified K-Fold** | K-Fold que preserva proporciones de clase |
| **LOOCV** | Validación cruzada con \(k = N\), un ejemplo de prueba por iteración |
| **Hiperparámetro** | Configuración externa que controla el proceso de aprendizaje |
| **Regularización** | Restricción o penalización que controla la complejidad del modelo |
| **Fuga de datos** | Uso indebido de información no disponible en inferencia real |
| **Score de Silhouette** | Métrica de clustering que compara cohesión interna y separación entre clústeres |

El **Score de Silhouette** pertenece al aprendizaje no supervisado: evalúa si un punto está más cerca de los elementos de su propio clúster que de los clústeres vecinos. No es una métrica de validación supervisada y no debe usarse para estimar generalización de clasificadores o regresores.

***

## Guía de Resolución Conceptual

### Procedimiento general de evaluación

```text
1. Definir problema, clase positiva y métrica de negocio.

2. Separar conjunto de prueba final.
   └─ Este conjunto no participa en selección ni ajuste.

3. Sobre el conjunto de desarrollo:
   a. Aplicar validación simple para prototipar.
   b. Aplicar Stratified K-Fold si es clasificación desbalanceada.
   c. Aplicar LOOCV sólo si N es pequeño y el entrenamiento es barato.

4. Diagnosticar:
   - Train alto + Test alto      → Underfitting
   - Train bajo + Test alto      → Overfitting
   - Train y Test cercanos       → Posible buen ajuste

5. Ajustar capacidad:
   - Underfitting  → más features, transformaciones, mayor complejidad
   - Overfitting   → regularización, más datos, menor complejidad

6. Optimizar hiperparámetros mediante búsqueda aleatoria
   con validación cruzada interna.

7. Evaluar una única vez en el test final.

8. Registrar dataset, folds, semillas, hiperparámetros y métricas.
```

### Validación de consistencia

Toda evaluación debe cumplir estas invariantes:

```text
Cada observación del conjunto de desarrollo:
    aparece exactamente una vez como validación en K-Fold.

En Stratified K-Fold:
    proporción de clases por fold ≈ proporción global.

En LOOCV:
    número de iteraciones = número de observaciones.

En búsqueda de hiperparámetros:
    ningún dato del test final participa en la selección.
```

### Corrección conceptual del material práctico

El material propone usar `RandomizedSearchCV`, `StratifiedKFold`, `LeaveOneOut` y `cross_val_score`; conceptualmente, el flujo es correcto, pero debe respetar una separación estricta entre:

1. **Conjunto de desarrollo**: usado para validación cruzada y búsqueda de hiperparámetros.
2. **Conjunto de prueba final**: usado sólo después de fijar el modelo y sus hiperparámetros.

La documentación de `scikit-learn` define K-Fold como \(k\) grupos, con \(k-1\) usados para entrenamiento y uno para validación; Stratified K-Fold preserva aproximadamente el porcentaje de cada clase; y LOOCV genera \(N\) conjuntos de entrenamiento y prueba, dejando una observación por iteración.  La búsqueda aleatoria muestrea candidatos desde distribuciones sobre el espacio de hiperparámetros bajo un presupuesto definido de iteraciones. [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)

| Etapa | Datos permitidos | Propósito |
|---|---|---|
| Entrenamiento de parámetros | Train fold | Aprender parámetros internos |
| Selección de hiperparámetros | Validación interna / folds | Elegir configuración |
| Estimación de generalización | Test final | Medir desempeño no sesgado |
| Monitoreo en producción | Datos nuevos | Detectar deriva y degradación |
