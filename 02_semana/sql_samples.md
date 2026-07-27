# ¿Qué es SQL?

SQL (Structured Query Language) es un lenguaje de consulta estructurado utilizado para interactuar con bases de datos relacionales. Se emplea para realizar operaciones como crear tablas, insertar, modificar y eliminar datos, así como consultar y analizar información.

## Principales características de SQL

- **Lenguaje declarativo:** Definimos el "qué" queremos hacer, no el "cómo".
- **Estándar:** Aunque diferentes sistemas de bases de datos (MySQL, PostgreSQL, SQLite, etc.) pueden tener pequeñas variaciones, SQL es un estándar comúnmente aceptado.
- **Versátil:** Permite consultar grandes volúmenes de datos de manera eficiente, lo que lo convierte en un pilar fundamental en la gestión de datos.

## Ejemplos de sistemas de gestión de bases de datos (SGBD) que utilizan SQL

- **MySQL:** Popular por su facilidad de uso y velocidad.
- **PostgreSQL:** Reconocido por ser altamente extensible y compatible con estándares.
- **SQLite:** Un motor de base de datos ligero y portátil.
- **Microsoft SQL Server:** Enfocado en soluciones empresariales con soporte de alta disponibilidad.
- **Oracle Database:** Ofrece soluciones avanzadas de escalabilidad y rendimiento.

## Casos de uso de SQL

- Gestión de inventarios.
- Aplicaciones financieras.
- Análisis de grandes volúmenes de datos.
- Sistemas de recomendación.

## Bases de Datos Relacionales

Las bases de datos relacionales organizan la información en tablas que contienen filas y columnas. Cada tabla se asocia a un modelo relacional, que define relaciones entre diferentes entidades del sistema.

### Conceptos clave en una base de datos relacional:

- **Tabla (Table):** Una colección de filas (registros) y columnas (campos).
- **Fila (Row):** Representa un único registro o instancia en la tabla.
- **Columna (Column):** Define un atributo o propiedad que cada fila tiene.
- **Clave Primaria (Primary Key):** Un identificador único para cada fila en una tabla.
- **Clave Foránea (Foreign Key):** Un campo que establece una relación entre dos tablas.

### Ejemplo de tabla Empleados

| id | nombre | departamento | salario |
|-----|---------------|------------------|----------|
| 1 | Juan Pérez | Ventas | 2500.50 |
| 2 | María Gómez | Marketing | 3000.00 |
| 3 | Carlos Ruiz | Desarrollo | 3200.00 |
| 4 | Laura Martín | Datos | 4000.00 |
| 5 | Pedro Salgado | Recursos Humanos | 2800.00 |
| 6 | Ana López | Datos | 4500.00 |
| 7 | Luis Sánchez | Recursos Humanos | 3100.00 |

### Relaciones entre tablas:

- **Uno a muchos (1:N):** Un registro de una tabla se relaciona con varios registros de otra.
- **Muchos a muchos (N:N):** Varios registros de una tabla se relacionan con varios registros de otra, utilizando una tabla intermedia.
- **Uno a uno (1:1):** Cada registro en una tabla se relaciona con un único registro en otra.

### Estructura básica de una tabla en SQL

Para crear una tabla en SQL, usamos la instrucción **CREATE TABLE**, especificando el nombre de la tabla y sus columnas, junto con los tipos de datos.

### Tipos de Datos Comunes en SQL

**Tipos Numéricos:**
- **INT:** Enteros. Ej: 1, 25, -10.
- **DECIMAL(precision, scale):** Números decimales. Ej: DECIMAL(10, 2) para 12345.67.
- **FLOAT:** Números en punto flotante.

**Tipos de Texto:**
- **VARCHAR(n):** Cadena de texto de longitud variable hasta un máximo de n caracteres.
- **CHAR(n):** Cadena de texto de longitud fija, siempre ocupa n caracteres.
- **TEXT:** Cadenas de texto más largas, útil para descripciones.

**Tipos de Fecha y Hora:**
- **DATE:** Almacena fechas en formato 'YYYY-MM-DD'.
- **TIME:** Almacena solo la hora en formato 'HH:MM:SS'.
- **DATETIME:** Combina fecha y hora.

**Tipos Lógicos:**
- **BOOLEAN:** Valores de verdadero o falso (en algunos SGBD puede representarse como TINYINT).

## Sublenguajes de SQL

### DDL (Data Definition Language) – Lenguaje de Definición de Datos

Este sublenguaje se usa para definir la estructura de las bases de datos, incluyendo la creación, modificación y eliminación de tablas, índices, y otros objetos de la base de datos.

**Principales comandos DDL:**
- **CREATE:** Crea objetos de la base de datos, como tablas, índices, etc.
- **ALTER:** Modifica la estructura de una tabla existente (añadir, modificar o eliminar columnas).
- **DROP:** Elimina objetos de la base de datos, como tablas o índices.
- **TRUNCATE:** Elimina todos los registros de una tabla de manera rápida, sin borrar su estructura.

### DML (Data Manipulation Language) – Lenguaje de Manipulación de Datos

Se utiliza para realizar operaciones CRUD (Create, Read, Update, Delete) sobre los datos almacenados en la base de datos.

**Principales comandos DML:**
- **SELECT:** Recupera datos de la base de datos.
- **INSERT:** Inserta nuevos registros en una tabla.
- **UPDATE:** Actualiza los datos existentes en una tabla.
- **DELETE:** Elimina registros específicos de una tabla.

### DCL (Data Control Language) – Lenguaje de Control de Datos

Se encarga de controlar los permisos y accesos a los objetos de la base de datos.

**Principales comandos DCL:**
- **GRANT:** Otorga permisos a un usuario o rol sobre los objetos de la base de datos.
- **REVOKE:** Revoca permisos previamente otorgados a un usuario o rol.

### TCL (Transaction Control Language) – Lenguaje de Control de Transacciones

Este sublenguaje gestiona las transacciones dentro de la base de datos.

**Principales comandos TCL:**
- **COMMIT:** Guarda permanentemente todos los cambios realizados en la transacción.
- **ROLLBACK:** Deshace todos los cambios realizados durante la transacción en curso.
- **SAVEPOINT:** Define puntos intermedios en una transacción para poder realizar un rollback parcial.
- **SET TRANSACTION:** Define las características de una transacción, como el nivel de aislamiento.
```

## Parte 2: Código SQL Organizado por Comandos

### Configuración Inicial y Conexión a Base de Datos

```python
import sqlite3
from google.colab import drive

# Montar Google Drive
drive.mount('/content/drive')

# Ruta donde se guardará la base de datos en Google Drive
db_path = '/content/drive/My Drive/ejemplo_sql2.db'

# Crear conexión a la base de datos en Google Drive
conexion = sqlite3.connect(db_path)
cursor = conexion.cursor()
```

### Comandos DDL (Data Definition Language)

#### CREATE TABLE - Crear tablas con relaciones

```sql
-- Crear tabla Departamentos
CREATE TABLE IF NOT EXISTS Departamentos (
    id_departamento INTEGER PRIMARY KEY,
    nombre_departamento TEXT NOT NULL
);

-- Crear tabla Empleados con clave foránea
CREATE TABLE IF NOT EXISTS Empleados (
    id_empleado INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    id_departamento INTEGER,
    salario REAL,
    fecha_contratacion TEXT,
    FOREIGN KEY (id_departamento) REFERENCES Departamentos(id_departamento)
);

-- Crear tabla Proyectos
CREATE TABLE IF NOT EXISTS Proyectos (
    id_proyecto INTEGER PRIMARY KEY,
    nombre_proyecto TEXT NOT NULL,
    id_departamento INTEGER,
    FOREIGN KEY (id_departamento) REFERENCES Departamentos(id_departamento)
);

-- Crear tabla Tareas con múltiples claves foráneas
CREATE TABLE IF NOT EXISTS Tareas (
    id_tarea INTEGER PRIMARY KEY,
    descripcion TEXT NOT NULL,
    id_proyecto INTEGER,
    id_empleado INTEGER,
    FOREIGN KEY (id_proyecto) REFERENCES Proyectos(id_proyecto),
    FOREIGN KEY (id_empleado) REFERENCES Empleados(id_empleado)
);
```

#### INSERT INTO - Insertar datos

```sql
-- Insertar datos en Departamentos
INSERT OR IGNORE INTO Departamentos (id_departamento, nombre_departamento) 
VALUES (1, "Ventas");
INSERT OR IGNORE INTO Departamentos (id_departamento, nombre_departamento) 
VALUES (2, "Marketing");
INSERT OR IGNORE INTO Departamentos (id_departamento, nombre_departamento) 
VALUES (3, "Desarrollo");

-- Insertar datos en Empleados
INSERT OR IGNORE INTO Empleados (id_empleado, nombre, id_departamento, salario, fecha_contratacion) 
VALUES (1, "Juan Pérez", 1, 2500.50, "2025-01-15");
INSERT OR IGNORE INTO Empleados (id_empleado, nombre, id_departamento, salario, fecha_contratacion) 
VALUES (2, "María Gómez", 2, 3000.00, "2025-01-16");
INSERT OR IGNORE INTO Empleados (id_empleado, nombre, id_departamento, salario, fecha_contratacion) 
VALUES (3, "Carlos Ruiz", 3, 3200.00, "2025-01-17");

-- Insertar datos en Proyectos
INSERT OR IGNORE INTO Proyectos (id_proyecto, nombre_proyecto, id_departamento) 
VALUES (1, "Proyecto A", 1);
INSERT OR IGNORE INTO Proyectos (id_proyecto, nombre_proyecto, id_departamento) 
VALUES (2, "Proyecto B", 2);

-- Insertar datos en Tareas
INSERT OR IGNORE INTO Tareas (id_tarea, descripcion, id_proyecto, id_empleado) 
VALUES (1, "Tarea 1 del Proyecto A", 1, 1);
INSERT OR IGNORE INTO Tareas (id_tarea, descripcion, id_proyecto, id_empleado) 
VALUES (2, "Tarea 2 del Proyecto A", 1, 3);
INSERT OR IGNORE INTO Tareas (id_tarea, descripcion, id_proyecto, id_empleado) 
VALUES (3, "Tarea del Proyecto B", 2, 2);
```

#### ALTER TABLE - Modificar estructura

```sql
-- Agregar una columna a una tabla existente
ALTER TABLE Empleados
ADD fecha_contratacion DATE;
```

#### UPDATE - Actualizar datos

```sql
-- Actualizar salario de un empleado específico
UPDATE Empleados 
SET salario = 2700.00 
WHERE id_empleado = 1;
```

#### DELETE - Eliminar registros

```sql
-- Eliminar un empleado específico
DELETE FROM Empleados 
WHERE id_empleado = 3;

-- Eliminar empleado con nombre específico
DELETE FROM Empleados 
WHERE nombre = 'Carlos Ruiz';
```

#### DROP TABLE - Eliminar tabla

```sql
-- Eliminar tabla completa
DROP TABLE Empleados;
```

### Comandos DML (Data Manipulation Language)

#### SELECT - Consultas básicas

```sql
-- Seleccionar todos los registros
SELECT * FROM Empleados;

-- Seleccionar columnas específicas
SELECT nombre, salario FROM Empleados;

-- Seleccionar con condición
SELECT * FROM Empleados 
WHERE id_empleado = 1;
```

#### SELECT con BETWEEN - Rango de valores

```sql
-- Filtrar salarios dentro de un rango
SELECT * FROM Empleados 
WHERE salario BETWEEN 2000 AND 3000;

-- Filtrar fechas dentro de un rango
SELECT * FROM Empleados 
WHERE fecha_contratacion BETWEEN '2025-01-01' AND '2025-01-31';
```

#### SELECT con IN - Múltiples valores

```sql
-- Filtrar por múltiples valores específicos
SELECT * FROM Departamentos 
WHERE nombre_departamento IN ('Ventas', 'Marketing');

-- Filtrar IDs específicos
SELECT * FROM Empleados 
WHERE id_departamento IN (1, 2, 3);
```

#### SELECT con IS NULL - Valores nulos

```sql
-- Buscar registros con valor nulo
SELECT * FROM Empleados 
WHERE fecha_contratacion IS NULL;

-- Buscar registros con valor no nulo
SELECT * FROM Empleados 
WHERE fecha_contratacion IS NOT NULL;
```

#### SELECT con LIKE - Patrones de texto

```sql
-- Buscar nombres que comienzan con 'Juan'
SELECT * FROM Empleados 
WHERE nombre LIKE 'Juan%';

-- Buscar nombres que terminan con 'ez'
SELECT * FROM Empleados 
WHERE nombre LIKE '%ez';

-- Buscar nombres que contienen 'a'
SELECT * FROM Empleados 
WHERE nombre LIKE '%a%';

-- Buscar nombres con un carácter específico en posición
SELECT * FROM Empleados 
WHERE nombre LIKE 'J_a%';
```

### Consultas con Múltiples Tablas (JOIN)

```sql
-- INNER JOIN: Empleados con su departamento
SELECT 
    e.nombre AS empleado,
    d.nombre_departamento AS departamento,
    e.salario
FROM Empleados e
INNER JOIN Departamentos d ON e.id_departamento = d.id_departamento;

-- JOIN con múltiples tablas
SELECT 
    e.nombre AS empleado,
    d.nombre_departamento AS departamento,
    p.nombre_proyecto AS proyecto,
    t.descripcion AS tarea
FROM Empleados e
INNER JOIN Departamentos d ON e.id_departamento = d.id_departamento
INNER JOIN Tareas t ON t.id_empleado = e.id_empleado
INNER JOIN Proyectos p ON t.id_proyecto = p.id_proyecto;
```

### Confirmación y Cierre de Conexión

```python
# Confirmar los cambios en la base de datos
conexion.commit()

# Cerrar la conexión
conexion.close()
```

### Consultas de Metadatos

```sql
-- Ver todas las tablas en la base de datos
SELECT name FROM sqlite_master WHERE type='table';

-- Ver estructura de una tabla específica
PRAGMA table_info(Empleados);
```