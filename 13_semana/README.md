# Evaluación de Clasificación: Matriz de Confusión y Métricas

La matriz de confusión es la representación canónica del comportamiento de un clasificador: descompone el resultado global en los cuatro tipos de aciertos y errores posibles. A partir de ella se derivan métricas como precisión, sensibilidad, especificidad y F1, cuya selección debe responder al costo asimétrico de los errores en el dominio del problema. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

***

## 13.1 Matriz de Confusión

### Fundamentos e internos

Una matriz de confusión es una tabla de contingencia que cruza la **etiqueta real** con la **etiqueta predicha**. En su convención más común, \(C_{i,j}\) contiene el número de observaciones cuya clase real es \(i\) y que el modelo predijo como \(j\); por ello, las filas suelen representar la verdad y las columnas la predicción. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

Para clasificación binaria, con la convención “positivo = 1” y “negativo = 0”, la estructura es:

```text
                         Predicción del modelo
                         Positivo (1)     Negativo (0)
                      ┌────────────────┬────────────────┐
   Clase real         │       TP       │       FN       │
   Positivo (1)       │ Verdadero      │ Falso          │
                      │ Positivo       │ Negativo       │
                      ├────────────────┼────────────────┤
   Clase real         │       FP       │       TN       │
   Negativo (0)       │ Falso          │ Verdadero      │
                      │ Positivo       │ Negativo       │
                      └────────────────┴────────────────┘
```

| Celda | Significado | Tipo de resultado | Error estadístico asociado |
|---|---|---|---|
| \(TP\) | Real positivo, predicho positivo | Acierto | — |
| \(TN\) | Real negativo, predicho negativo | Acierto | — |
| \(FP\) | Real negativo, predicho positivo | Error | Error Tipo I: falsa alarma |
| \(FN\) | Real positivo, predicho negativo | Error | Error Tipo II: omisión del caso positivo |

La matriz no es una métrica: es la **fuente estructural** de métricas. Su valor arquitectónico está en que preserva la evidencia de *qué tipo* de error ocurre, en lugar de colapsar todo el desempeño en un único número. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### Por qué Accuracy puede engañar

El *Accuracy* mide la proporción global de aciertos:

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

En un conjunto fuertemente desbalanceado, un clasificador trivial que predice siempre la clase mayoritaria puede obtener un *Accuracy* alto sin detectar la clase de interés. Por ejemplo, si el 99% de los registros son negativos, predecir siempre “negativo” produce 99% de exactitud y 100% de falsos negativos para la clase positiva. Por esta razón, en contextos desbalanceados Accuracy y Precision pueden favorecer a la clase mayoritaria y subrepresentar los resultados minoritarios. [link.springer](https://link.springer.com/article/10.1186/s40537-025-01313-4)

### Flujo metodológico de cálculo

```text
1. Reservar conjunto de prueba
   └─ Datos no utilizados en entrenamiento ni ajuste de hiperparámetros.

2. Generar predicciones
   └─ Aplicar el modelo entrenado a las características de prueba.

3. Comparar pares (y_real, y_pred)
   └─ Asignar cada observación a TP, TN, FP o FN.

4. Tabular conteos
   └─ Construir la matriz de confusión.

5. Interpretar errores según negocio
   └─ Determinar si domina el costo de FP o de FN.

6. Derivar métricas
   └─ Calcular Accuracy, Precision, Recall, Specificity, F1, Error Rate.
```

Pseudocódigo formal, independiente de cualquier framework:

```text
función construir_matriz_confusión(y_real, y_pred):
    TP ← 0
    TN ← 0
    FP ← 0
    FN ← 0

    para cada par (r, p) en zip(y_real, y_pred):
        si r = 1 y p = 1 entonces TP ← TP + 1
        sino si r = 1 y p = 0 entonces FN ← FN + 1
        sino si r = 0 y p = 1 entonces FP ← FP + 1
        sino TN ← TN + 1

    devolver matriz(TP, FN, FP, TN)
```

### Patrones de diseño y arquitectura

En sistemas de ML en producción, la matriz de confusión debe tratarse como un **artefacto de observabilidad**, no solo como un resultado de notebook.

- **Evaluación offline**: se calcula sobre un conjunto de prueba inmutable, con partición estratificada cuando las clases están desbalanceadas.
- **Evaluación por segmentos**: además de la matriz global, conviene calcular matrices por región, canal, versión de modelo, ventana temporal o cohorte de usuarios.
- **Monitoreo post-despliegue**: las tasas de FP y FN deben registrarse como señales operativas, especialmente cuando el modelo dispara acciones automáticas.
- **Versionado de evaluación**: cada matriz debe asociarse a una versión de dataset, versión de modelo, umbral de decisión y definición explícita de clase positiva.
- **Separación de responsabilidades**: el servicio de inferencia produce probabilidades o etiquetas; un componente de evaluación calcula métricas y almacena evidencia auditable.

```text
Datos etiquetados ──► Partición de prueba ──► Modelo versionado
                                                     │
                                                     ▼
                                          Predicciones + probabilidades
                                                     │
                                                     ▼
                                     Evaluador de métricas / matriz
                                                     │
                       ┌─────────────────────────────┼─────────────────────────────┐
                       ▼                             ▼                             ▼
              Reporte de calidad           Alertas de deriva / umbral       Auditoría y trazabilidad
```

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Definir explícitamente cuál clase es “positiva”; en detección de fraude, enfermedad o spam, la clase positiva debe ser la acción o condición de interés.
- Calcular la matriz sobre datos de prueba verdaderamente no vistos; evaluar sobre datos de entrenamiento infla artificialmente el desempeño.
- Reportar siempre el umbral de decisión junto con la matriz, porque TP, FP, TN y FN dependen de ese umbral.
- Estratificar la partición de prueba cuando el desbalance es relevante.
- Documentar el costo relativo de un FP frente a un FN antes de elegir la métrica principal.

**Anti-patrones**

| Anti-patrón | Riesgo | Corrección |
|---|---|---|
| Elegir Accuracy como única métrica | Oculta fallos graves en la clase minoritaria | Usar matriz + Precision/Recall/F1 según costo |
| Confundir filas y columnas | Invierte la interpretación de FP y FN | Fijar y documentar la convención: filas = real, columnas = predicción |
| Evaluar con datos de entrenamiento | Sobreestima el rendimiento | Usar conjunto de prueba aislado |
| Optimizar F1 sin análisis de negocio | Puede ser inadecuado si FP y FN tienen costos muy distintos | Usar métrica o función de costo alineada al dominio |
| Cambiar el umbral sin registrar su efecto | La matriz deja de ser comparable entre ejecuciones | Versionar umbral y matriz por evaluación |
| Ignorar el soporte de cada clase | Métricas altas pueden provenir de muestras diminutas | Reportar conteos y tamaño de clase junto a las métricas |

***

## 13.2 Métricas Derivadas

### Fundamentos e internos

Las métricas binarias responden a preguntas distintas sobre la misma matriz. La precisión evalúa la confiabilidad de las alertas positivas; la sensibilidad evalúa la cobertura de los positivos reales; la especificidad evalúa la cobertura de los negativos reales. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

| Métrica | Fórmula | Pregunta que responde | Se prioriza cuando |
|---|---|---|---|
| Accuracy | \(\frac{TP+TN}{TP+TN+FP+FN}\) | ¿Qué proporción total de decisiones fue correcta? | Clases aproximadamente balanceadas y errores simétricos |
| Precision | \(\frac{TP}{TP+FP}\) | De las alertas positivas, ¿cuántas fueron correctas? | Un FP es costoso |
| Recall / Sensibilidad | \(\frac{TP}{TP+FN}\) | De los positivos reales, ¿cuántos fueron detectados? | Un FN es costoso |
| Specificity | \(\frac{TN}{TN+FP}\) | De los negativos reales, ¿cuántos fueron correctamente descartados? | Los FP deben minimizarse |
| F1 | \(2 \cdot \frac{P \cdot R}{P+R}\) | ¿Qué equilibrio existe entre precisión y sensibilidad? | Se requiere un resumen único con desbalance |
| Error Rate | \(\frac{FP+FN}{TP+TN+FP+FN}\) | ¿Qué proporción global de decisiones fue incorrecta? | Complemento directo de Accuracy |

La precisión es \(TP/(TP+FP)\) y la sensibilidad es \(TP/(TP+FN)\); ambas se derivan directamente de la matriz de confusión.  En clasificación binaria, el recall de la clase positiva se conoce como sensibilidad, mientras que el recall de la clase negativa corresponde a la especificidad. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

### Trade-off central: Precision vs. Recall

Existe una tensión estructural: hacer el criterio de decisión más permisivo suele aumentar el recall, pero puede reducir la precisión; hacerlo más estricto suele aumentar la precisión, pero puede dejar pasar más positivos reales.

```text
Umbral bajo / decisión permisiva
        │
        ▼
Más casos marcados como positivos
        │
        ├── Recall tiende a subir
        └── Precision puede bajar por más FP

Umbral alto / decisión estricta
        │
        ▼
Menos casos marcados como positivos
        │
        ├── Precision tiende a subir
        └── Recall puede bajar por más FN
```

| Escenario | Error dominante | Métrica prioritaria | Ejemplo |
|---|---|---|---|
| Diagnóstico de enfermedad grave | FN: paciente enfermo no detectado | Recall / Sensibilidad | Cribado médico |
| Detección de fraude con revisión manual costosa | FP: transacción legítima bloqueada | Precision | Antifraude financiero |
| Filtrado de spam | FP: correo legítimo marcado como spam | Precision | Gestión de bandeja de entrada |
| Detección de intrusión crítica | FN: ataque no detectado | Recall / Sensibilidad | Seguridad |
| Sistema de triage con ambos errores críticos | FP y FN relevantes | F1 o función de costo | Clasificación clínica asistida |

### F1: media armónica, no media aritmética

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Equivalentemente:

\[
F1 = \frac{2TP}{2TP + FP + FN}
\]

La media armónica penaliza el desequilibrio: si una de las dos métricas es muy baja, F1 también será bajo. Por ejemplo, con Precision \(= 1.0\) y Recall \(= 0.1\), la media aritmética sería \(0.55\), pero F1 sería aproximadamente \(0.182\). El F1 balancea por igual precisión y recall, y es especialmente útil en conjuntos desbalanceados. [link.springer](https://link.springer.com/article/10.1186/s40537-025-01313-4)

### Caso práctico: spam

Datos documentados:

\[
TP = 85,\quad FN = 15,\quad FP = 5,\quad TN = 95,\quad N = 200
\]

Matriz resultante:

```text
                         Predicción
                         Spam (1)    No Spam (0)
                      ┌────────────┬────────────┐
   Real Spam (1)      │    85      │    15      │
                      ├────────────┼────────────┤
   Real No Spam (0)   │     5      │    95      │
                      └────────────┴────────────┘
```

Cálculos:

\[
\text{Accuracy} = \frac{85 + 95}{200} = 0.900
\]

\[
\text{Precision} = \frac{85}{85 + 5} = 0.944
\]

\[
\text{Recall} = \frac{85}{85 + 15} = 0.850
\]

\[
F1 = 2 \times \frac{0.944 \times 0.850}{0.944 + 0.850} \approx 0.894
\]

\[
\text{Error Rate} = \frac{5 + 15}{200} = 0.100
\]

| Métrica | Valor | Lectura técnica |
|---|---:|---|
| Accuracy | 90.0% | 180 de 200 correos fueron clasificados correctamente |
| Precision | 94.4% | De 90 correos marcados como spam, 85 eran realmente spam |
| Recall | 85.0% | De 100 correos realmente spam, se detectaron 85 |
| F1 | 89.4% | Buen equilibrio entre no marcar spam legítimo y capturar spam real |
| Error Rate | 10.0% | 20 errores totales: 5 FP y 15 FN |

En este dominio, el costo de un FP suele ser mayor que el de un FN: enviar un correo legítimo a spam puede causar pérdida de comunicación importante. Por ello, la precisión de 94.4% es especialmente valiosa, aunque los 15 FN siguen siendo relevantes para la cobertura del filtro. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

### Caso práctico: diagnóstico médico

Datos:

\[
TP = 50,\quad FP = 10,\quad TN = 40,\quad FN = 5
\]

\[
\text{Accuracy} = \frac{50 + 40}{105} \approx 0.857
\]

\[
\text{Precision} = \frac{50}{50 + 10} \approx 0.833
\]

\[
\text{Recall} = \frac{50}{50 + 5} \approx 0.909
\]

\[
\text{Specificity} = \frac{40}{40 + 10} = 0.800
\]

\[
F1 = 2 \times \frac{0.833 \times 0.909}{0.833 + 0.909} \approx 0.869
\]

Aquí los \(FN = 5\) representan pacientes enfermos no detectados, el riesgo más severo. La prioridad de diseño sería maximizar la sensibilidad, aceptando potencialmente más falsos positivos que luego puedan resolverse mediante pruebas confirmatorias. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

### Patrones de diseño y arquitectura

| Patrón | Aplicación | Beneficio |
|---|---|---|
| **Métrica norte única** | Definir una métrica principal alineada al negocio | Evita optimizar un número sin impacto real |
| **Matriz por segmento** | Calcular TP/FP/TN/FN por cohorte | Detecta fallos ocultos en subpoblaciones |
| **Evaluación por umbral** | Analizar Precision y Recall para distintos umbrales | Permite elegir el punto operativo adecuado |
| **Función de costo** | Asignar pesos económicos a FP y FN | Traduce errores técnicos a impacto de negocio |
| **Reporte multiclase** | Extender la matriz a \(k \times k\) clases | Identifica qué clases se confunden entre sí |
| **Auditoría de evaluación** | Guardar dataset, modelo, umbral y matriz | Habilita reproducibilidad y trazabilidad |

En problemas multiclase, la matriz se expande a \(k \times k\): la diagonal contiene aciertos y las celdas fuera de la diagonal revelan confusiones específicas entre clases. Las métricas por clase pueden agregarse mediante promedios macro, micro o ponderado, pero la elección del promedio cambia sustancialmente la interpretación. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### Tips prácticos y anti-patrones

**Buenas prácticas**

- Reportar siempre al menos: matriz, tamaño de muestra, clase positiva, umbral y métricas principales.
- En dominios de alto riesgo, complementar F1 con análisis de curvas Precision-Recall y con una función de costo explícita.
- Usar Specificity cuando el problema exige descartar correctamente casos negativos, no solo detectar positivos.
- Validar la estabilidad de las métricas mediante validación cruzada o múltiples particiones, especialmente con clases minoritarias pequeñas.
- Tratar las métricas como requisitos de calidad medibles: por ejemplo, “Recall mínimo de 0.95 en producción para la clase crítica”.

**Anti-patrones**

| Anti-patrón | Consecuencia | Mitigación |
|---|---|---|
| Usar F1 como métrica universal | Ignora costos asimétricos entre FP y FN | Definir métrica según dominio o función de costo |
| Reportar Precision sin Recall | Un modelo puede ser preciso pero detectar muy pocos positivos | Reportar ambas y su relación |
| Reportar Recall sin Precision | Un modelo puede detectar casi todo generando muchas falsas alarmas | Reportar ambas y su relación |
| Comparar modelos con umbrales distintos | Comparación inválida | Fijar o reportar el umbral de decisión |
| Ocultar el soporte de clase | Métricas aparentemente buenas sobre muestras mínimas | Incluir conteos por clase |
| Interpretar Accuracy en datos desbalanceados | falsa sensación de calidad | Priorizar matriz, Precision, Recall y F1 |

***

## Guía de resolución conceptual

### Procedimiento general

```text
Entrada:
    y_real  : etiquetas verdaderas
    y_pred  : etiquetas predichas
    positiva: clase considerada positiva

Salida:
    matriz_confusión, métricas

Proceso:
    1. Verificar que y_real y y_pred tengan igual longitud.
    2. Verificar que ambas secuencias contengan las mismas clases válidas.
    3. Para cada observación:
         - TP si real = positiva y predicción = positiva
         - FN si real = positiva y predicción = negativa
         - FP si real = negativa y predicción = positiva
         - TN si real = negativa y predicción = negativa
    4. Construir la matriz.
    5. Calcular métricas.
    6. Interpretar FP y FN según el costo del dominio.
```

### Validación de consistencia

Antes de aceptar una matriz, deben cumplirse estas invariantes:

\[
TP + FN = \text{total de positivos reales}
\]

\[
FP + TN = \text{total de negativos reales}
\]

\[
TP + FP = \text{total de predicciones positivas}
\]

\[
FN + TN = \text{total de predicciones negativas}
\]

\[
TP + TN + FP + FN = N
\]

Para el caso de spam:

```text
Positivos reales  = TP + FN = 85 + 15  = 100
Negativos reales  = FP + TN = 5 + 95   = 100
Predicciones pos. = TP + FP = 85 + 5   = 90
Predicciones neg. = FN + TN = 15 + 95  = 110
Total             = 200
```

Estas comprobaciones previenen errores comunes de orientación de matriz, selección incorrecta de clase positiva o reconstrucción defectuosa de los vectores de evaluación. La definición estándar de matriz de confusión exige que cada celda combine clase real y clase predicha de forma consistente. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

***

## Nota sobre la implementación solicitada

El material incluye una implementación en Python con `scikit-learn`, `pandas`, `matplotlib` y `seaborn`; sin embargo, contiene expresiones incompletas como `labels=`, `cm` sin índices y asignaciones incompletas para `tp`, `fn`, `fp` y `tn`. Bajo la regla de atemporalidad establecida, la solución correcta debe expresarse mediante pseudocódigo formal y no mediante sintaxis dependiente de librerías.

La reconstrucción conceptual correcta del caso spam es:

```text
y_real:
    85 valores = 1   (TP)
    15 valores = 1   (FN)
     5 valores = 0   (FP)
    95 valores = 0   (TN)

y_pred:
    85 valores = 1   (TP)
    15 valores = 0   (FN)
     5 valores = 1   (FP)
    95 valores = 0   (TN)
```

Con filas = clase real y columnas = predicción, la matriz debe ser:

```text
              Predicho Spam   Predicho No Spam
Real Spam          85                15
Real No Spam        5                95
```

Por tanto, los valores esperados son:

| Métrica | Valor esperado |
|---|---:|
| Accuracy | 0.9000 |
| Precision | 0.9444 |
| Recall | 0.8500 |
| F1 | 0.8947 |
| Error Rate | 0.1000 |

La biblioteca de métricas de `scikit-learn` expone funciones específicas para matriz de confusión, Accuracy, Precision, Recall y F1, además de reportes por clase; conceptualmente, todas ellas se reducen a los conteos de la matriz. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
