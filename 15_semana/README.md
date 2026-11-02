# Metodos de ensamble
El concepto central de los **Modelos de Ensamble** es muy intuitivo: un comité toma mejores decisiones que una sola persona. Al combinar varios algoritmos, logramos compensar sus debilidades individuales para obtener predicciones mucho más precisas y estables frente a los clásicos problemas de sesgo (aprender muy poco) y varianza (memorizar demasiado).

![Metodos de ensamble](./.img/metodos_ensamble.jpeg)

### Los 3 Enfoques Clave de los Ensambles

* **Bagging (Trabajo en paralelo):** Entrena muchos modelos al mismo tiempo utilizando distintas porciones aleatorias de los datos originales. La decisión final se toma por promedio (en regresión) o por votación mayoritaria (en clasificación).
* *El objetivo:* Reducir la **varianza** para ganar muchísima estabilidad.
* *Analogía:* Preguntar a 100 personas distintas cuántos caramelos hay en un frasco y promediar sus respuestas.
* *Modelo estrella:* **Random Forest**.


* **Boosting (Aprendizaje secuencial):** Entrena modelos débiles uno detrás de otro. La magia ocurre porque cada nuevo modelo se enfoca exclusivamente en **corregir los errores** que dejó el modelo anterior.
* *El objetivo:* Reducir el **sesgo** permitiendo construir fronteras de decisión progresivamente más complejas.
* *Analogía:* Un estudiante que, tras reprobar un examen, en su siguiente intento estudia únicamente las preguntas en las que falló.
* *Modelos estrella:* **Gradient Boosting**, **XGBoost**, **LightGBM** y **AdaBoost**.


* **Stacking (El gerente del comité):** Entrena varios modelos que son completamente distintos entre sí (por ejemplo, regresiones combinadas con árboles). Luego, utiliza un **meta-modelo** superior que toma las predicciones de estos modelos base y aprende matemáticamente en cuál confiar más según el caso.
* *El objetivo:* Aprovechar fortalezas heterogéneas.
* *Analogía:* Un líder de equipo que escucha al experto financiero, al de marketing y al de sistemas, y aprende a qué especialista hacerle más caso dependiendo del problema.



### El Freno de Mano: La Regularización

Especialmente al usar técnicas potentes como Boosting, el algoritmo puede volverse tan complejo que termina sobreajustándose (aprendiendo el "ruido" de los datos). Para evitarlo, usamos la **Regularización**, una técnica que directamente **penaliza la complejidad**:

* **Lasso (L1):** Es drástico. Ayuda a seleccionar variables porque apaga por completo (lleva a cero) el peso de los datos que no sirven.
* **Ridge (L2):** Es equitativo. Encoge el peso de las variables para evitar que unas pocas dominen exageradamente la decisión.
* **Elastic Net:** El punto medio que combina lo mejor de ambas estrategias.

### La Regla de Oro en Evaluación

No asumas que un ensamble complejo siempre es mejor. El estándar correcto es comenzar con un modelo base muy simple, ajustar la configuración de los ensambles mediante **validación cruzada** (usando predicciones *out-of-fold* para no hacer trampa) y reservar estrictamente un conjunto de datos ciego (Test Set) para una única evaluación final honesta.

---

**Takeaway:** Los ensambles transforman modelos inestables en herramientas de predicción robustas: Bagging promedia en paralelo para dar estabilidad, Boosting corrige errores en serie para dar precisión, y Stacking usa un meta-modelo para exprimir lo mejor de diferentes algoritmos.

### XGBoost vs LightGBM

Aunque ambos dominan el ecosistema de datos tabulares estructurados mediante Gradient Boosting, sus diferencias arquitectónicas en la construcción de los árboles y la gestión de memoria definen su rendimiento en producción.

La diferencia más fundamental radica en cómo expanden los árboles de decisión:

### Diferencias Algorítmicas Clave

* **Estrategia de Crecimiento:**
* **XGBoost (Level-wise / Depth-wise):** Crece el árbol nivel por nivel. Calcula las divisiones para todos los nodos de la misma profundidad antes de pasar al siguiente nivel. Esto produce árboles simétricos y es inherentemente conservador, lo que ayuda a prevenir el sobreajuste.
* **LightGBM (Leaf-wise / Best-first):** Expande únicamente la hoja que maximiza la reducción de la función de pérdida (delta loss), sin importar el nivel en el que se encuentre. Genera árboles asimétricos y profundos. Alcanza menor pérdida más rápido, pero es altamente propenso al sobreajuste en datasets pequeños si no se restringe explícitamente la profundidad (`max_depth`) o las muestras por hoja (`min_data_in_leaf`).


* **Algoritmo de División (Split Finding):**
* **XGBoost:** Históricamente utilizaba un algoritmo de pre-ordenamiento exacto (*Exact Greedy Algorithm*), que evalúa todos los puntos de corte posibles. Para escalar, adoptó una aproximación basada en histogramas en versiones recientes (`tree_method='hist'`), pero su arquitectura base penaliza la memoria.
* **LightGBM:** Fue diseñado nativamente en torno al binning por histogramas. Discretiza los valores continuos en *bins* discretos (por defecto 255) antes del entrenamiento. Esto reduce drásticamente el costo computacional de encontrar el *split* $O(\text{datos} \times \text{features})$ a $O(\text{bins} \times \text{features})$ y minimiza la huella de RAM.


* **Muestreo y Manejo de Features:**
* **LightGBM (GOSS y EFB):** Implementa *Gradient-based One-Side Sampling* (GOSS), que retiene las instancias con gradientes grandes (casos donde el modelo falla) y submuestrea aleatoriamente las de gradientes pequeños, escalando el peso para no alterar la distribución. Además, usa *Exclusive Feature Bundling* (EFB) para agrupar variables fuertemente dispersas (sparse) de forma mutuamente excluyente, reduciendo la dimensionalidad sin pérdida de información.
* **XGBoost:** Carece de GOSS o EFB nativos, apoyándose en submuestreos aleatorios estándar de columnas y filas y heurísticas para matrices dispersas (Sparsity-aware Split Finding).



### Diferencias en Manejo de Datos

| Característica | XGBoost | LightGBM |
| --- | --- | --- |
| **Variables Categóricas** | Soporte nativo añadido recientemente, pero tradicionalmente requería *One-Hot Encoding*, penalizando árboles profundos al diluir la ganancia de información. | Soporte nativo altamente optimizado. Agrupa categorías buscando la división óptima mediante el método de Fisher, superando ampliamente al OHE en dimensionalidad alta. |
| **Consumo de Memoria** | Alto. Si se usa el método exacto, guarda valores y punteros pre-ordenados, consumiendo múltiples veces el tamaño del dataset en RAM. | Muy bajo. Los histogramas se almacenan como enteros de 8-bit (uint8), reduciendo drásticamente la huella en memoria. |
| **Velocidad de Entrenamiento** | Referencia estándar, pero computacionalmente pesado en datasets volumétricos. | Entre 2x y 10x más rápido que XGBoost en datasets masivos debido al binning y GOSS. |

### Cuándo usar cada uno

**Elige LightGBM cuando:**

1. Tienes un volumen de datos considerable (generalmente $>10,000$ registros; en datasets muy chicos, el enfoque *leaf-wise* memoriza el ruido).
2. Tienes latencia estricta en el pipeline de reentrenamiento o recursos de cómputo limitados.
3. El dataset tiene muchas variables categóricas de alta cardinalidad.

**Elige XGBoost cuando:**

1. Trabajas con datasets pequeños a medianos donde la prioridad es la máxima generalización y evitar el sobreajuste.
2. Buscas extrema estabilidad frente a hiperparámetros por defecto (su crecimiento *level-wise* perdona más configuraciones subóptimas).
3. Estás construyendo un sistema donde la interpretabilidad de la estructura del árbol es fundamental, ya que los árboles simétricos son más predecibles matemáticamente.
