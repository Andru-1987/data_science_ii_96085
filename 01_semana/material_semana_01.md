# Resumen de Fundamentos de Bases de Datos - Clase 1

## 1. Definición y Conceptos Fundamentales

### ¿Qué es una base de datos?
Una base de datos se define como una colección organizada de información estructurada o datos, normalmente almacenados electrónicamente en un sistema informático. El objetivo principal es estandarizar procesos y lograr mayor eficiencia en el manejo de la información.

### Definición mejorada
La base de datos es un conjunto de datos no redundantes, almacenados en un soporte informático, organizados en forma independiente de su utilización y accesibles simultáneamente por distintos usuarios y aplicaciones.

### Características importantes
- **No redundancia**: Eficiencia en el almacenamiento
- **Soporte informático**: Necesidad de medios físicos que garanticen seguridad
- **Independencia de uso**: No se necesita saber cómo están almacenados para acceder a los datos
- **Acceso simultáneo**: Flexibilidad para múltiples usuarios

---

## 2. Mejoras Ofrecidas por las Bases de Datos

El Sistema de Gestión de Bases de Datos (DBMS) proporciona las siguientes mejoras:

### Independencia
Separación entre la representación lógica de los datos y su almacenamiento físico.

### Integridad
Diversos niveles de protección frente a fallos y corrupción de datos.

### Eficiencia
Técnicas específicas para acelerar consultas y utilización de datos.

### Seguridad
Sistema complejo para otorgar permisos y garantizar acceso controlado.

### Centralización
Administración centralizada de los datos.

### Reusabilidad
Situaciones comunes de acceso y uso compartidas entre distintas aplicaciones.

### Acceso concurrente
Mecanismos para evitar corrupción de datos cuando múltiples usuarios acceden simultáneamente.

---

## 3. Sistemas de Gestión de Bases de Datos (DBMS)

### Definición
El DBMS es una solución tecnológica utilizada para optimizar y administrar el almacenamiento y recuperación de datos. Es un software específicamente diseñado para definir, manipular y utilizar la información contenida en las bases de datos.

### Funciones principales
- Administración del almacenamiento
- Recuperación de datos
- Mantenimiento de integridad y seguridad
- Interfaz para usuarios

---

## 4. Tipos de Bases de Datos

### Bases de Datos SQL (Relacionales)
- Basadas en el modelo de datos relacional propuesto por EF Codd en 1970
- Almacenan datos en forma de filas (tuplas) y columnas (atributos)
- Forman tablas (relaciones)
- Utilizan SQL para almacenar, manipular y mantener datos
- Ejemplos: MySQL, Microsoft SQL Server, Oracle

### Bases de Datos NoSQL (No Relacionales)
- Propuestas por Carlo Strozzi en 1998
- No almacenan datos únicamente en forma tabular
- Surgieron por la demanda de aplicaciones modernas
- Subtipos principales:
  - **Key-value**: Riak, Redis
  - **Documentos**: CouchDB, MongoDB
  - **Grafos**: Neo4j, GraphDB
  - **Wide-column**: Cassandra, HBase

### NewSQL
- Combinación de escalabilidad de NoSQL
- Garantiza estructura ACID de RDBMS

### Analytical
- Almacena y administra big data usando Business Intelligence (BI)

### As a service
- Servidor que utiliza una aplicación de base de datos que proporciona servicios a otros programas

---

## 5. Ventajas y Desventajas Comparativas

### Bases de Datos SQL (Relacionales)

| Ventajas | Desventajas |
|----------|-------------|
| Simplicidad del modelo | Mantenimiento difícil por acumulación de datos |
| Fácil uso y recuperación | Costos fijos y variables de mantenimiento |
| Precisión y organización | Requiere mucha memoria física |
| Integridad de datos | Poca escalabilidad en diferentes servidores |
| Normalización | Estructura compleja limitada a forma tabular |
| Colaboración multiusuario | Reducción de performance en tiempo |
| Integridad y seguridad | Mayor complejidad = menor tiempo de respuesta |

### Bases de Datos NoSQL (No Relacionales)

| Ventajas | Desventajas |
|----------|-------------|
| Modelo flexible (estructurados y no estructurados) | Falta de estandarización |
| Modelo de datos en evolución dinámica | Problemas de backup |
| Fácil escalamiento | Consistencia menos eficiente |
| Alto performance y baja latencia | Difícil mantenimiento y personal especializado |
| Acceso libre sin licencias costosas | Poco nivel de madurez |

---

## 6. Funciones y Lenguajes de Bases de Datos

### DDL (Data Definition Language)
- **CREATE**: Crear nueva base de datos u objetos
- **ALTER**: Reestructurar tablas (columnas, tipos, índices)
- **DROP**: Eliminar objetos de la base de datos
- **TRUNCATE**: Vaciar una tabla (datos fuera, estructura queda)
- **RENAME**: Cambiar nombre de objetos
- **COMMENT**: Agregar comentarios en el diccionario de datos

### DML (Data Manipulation Language)
- **SELECT**: Seleccionar valores de columnas
- **INSERT**: Insertar nuevos registros
- **UPDATE**: Actualizar registros existentes
- **DELETE**: Eliminar registros
- **LOCK**: Bloquear privilegios de lectura/escritura
- **MERGE**: Fusionar registros de una tabla

### DCL (Data Control Language)
- **GRANT**: Otorgar derechos o privilegios a usuarios
- **REVOKE**: Retirar derechos o privilegios de usuarios

### TCL (Transaction Control Language)
- **COMMIT**: Confirmar transacción
- **ROLLBACK**: Rehacer cambios por error
- **SAVEPOINT**: Agregar punto de control al proceso
- **SET TRANSACTION**: Definir características de una transacción

---

## 7. Glosario de Términos Clave

| Término | Definición |
|---------|------------|
| Base de datos | Colección ordenada de datos para estandarizar procesos y ser más eficientes |
| Backup | Sistema para guardar periódicamente información almacenada y evitar pérdidas |
| DBMS | Proceso para optimizar y administrar el almacenamiento y recuperación de datos |
| Mecanismos de conexión | Parámetros que permiten interactuar con las bases de datos |
| Arquitectura de 3 capas | Representación del sistema DBMS en componentes: externo, lógico e interno |
| Auditoría | Proceso para mejorar estructuras y registrar logs y acciones de usuarios |

---

## Apunte del Docente: Data Science II

### Contextualización para el Estudiante

En el marco de nuestra materia de Data Science II, es fundamental comprender que las bases de datos son el pilar sobre el cual construimos cualquier proyecto de análisis y modelado de datos. Antes de aplicar técnicas avanzadas de machine learning, debemos dominar los fundamentos de almacenamiento y gestión de datos.

### Relación con el Proyecto Final

Para su proyecto final, donde entrenarán y optimizarán modelos de machine learning, deberán considerar:

1. **Elección del tipo de base de datos**: Dependiendo del problema de negocio, podrían necesitar una base SQL para datos estructurados o NoSQL para datos no estructurados o semiestructurados.

2. **Integridad de datos**: Antes de entrenar modelos, aseguren que sus datos cumplan con los criterios de integridad (no redundancia, consistencia).

3. **Escalabilidad**: Consideren el crecimiento potencial de los datos al seleccionar el sistema de base de datos.

4. **Performance**: Para modelos que requieren consultas frecuentes, la eficiencia de la base de datos impacta directamente en el tiempo de entrenamiento.

### Recomendaciones Prácticas

1. **Backups**: Establezcan un sistema regular de backup de sus datasets, especialmente cuando trabajen con datos que serán transformados para el proyecto.

2. **Lenguajes**: Familiarícense al menos con los comandos básicos de SQL (SELECT, INSERT, UPDATE, DELETE) ya que serán herramientas fundamentales en su carrera como data scientists.

3. **Documentación**: Mantengan una auditoría de los cambios realizados en los datasets (logs, versiones, transformaciones aplicadas).

4. **Seguridad**: Si trabajan con datos sensibles, implementen los permisos adecuados utilizando DCL.

### Reflexión Final

La elección entre bases SQL y NoSQL no es una decisión técnica aislada, sino que debe alinearse con los objetivos del proyecto, el tipo de datos disponibles y los requisitos de escalabilidad y performance. En Data Science, la calidad de nuestros modelos depende en gran medida de la calidad y accesibilidad de los datos, por lo que estos fundamentos son el primer paso hacia un proyecto exitoso.