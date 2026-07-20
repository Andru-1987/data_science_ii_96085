"""
Introduccion a SQLite con Python
=================================


1. SQLite es una base de datos SQL sin servidor: no requiere instalar ni
   levantar un servicio aparte, lee y escribe en un unico archivo local.
2. El modulo `sqlite3` viene incluido en la libreria estandar de Python,
   no hace falta instalar nada extra para conectarse.
3. Flujo tipico de trabajo:
   - Crear una conexion (`create_connection`).
   - Ejecutar sentencias DDL/DML (`CREATE TABLE`, `INSERT`) con
     `execute_query`.
   - Leer datos (`SELECT`, incluyendo `JOIN`) con `execute_read_query`.
4. Modelo de datos de ejemplo (blog simple): users -> posts -> comments/likes,
   relacionados por claves foraneas (`user_id`, `post_id`).

"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from sqlite3 import Connection, Error

DB_PATH = Path(__file__).parent / "blog.sqlite"


def create_connection(path: Path) -> Connection | None:
    """Abre (o crea) el archivo de base de datos SQLite en `path`."""
    try:
        connection = sqlite3.connect(path)
        print(f"Conexion a SQLite DB exitosa ({path})")
        return connection
    except Error as e:
        print(f"El error '{e}' ha ocurrido")
        return None


def execute_query(connection: Connection, query: str) -> None:
    """Ejecuta una sentencia sin retorno de filas (CREATE, INSERT, UPDATE...)."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
    except Error as e:
        print(f"El error '{e}' ha ocurrido")


def execute_read_query(connection: Connection, query: str) -> list[tuple]:
    """Ejecuta un SELECT y devuelve todas las filas."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"El error '{e}' ha ocurrido")
        return []


CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        nationality TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        post_id INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (post_id) REFERENCES posts (id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS likes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        post_id INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (post_id) REFERENCES posts (id)
    );
    """,
]

SEED_DATA = [
    """
    INSERT INTO users (name, age, gender, nationality)
    VALUES
        ('James', 25, 'hombre', 'USA'),
        ('Leila', 32, 'mujer', 'France'),
        ('Brigitte', 35, 'mujer', 'England'),
        ('Mike', 40, 'hombre', 'Denmark'),
        ('Elizabeth', 21, 'mujer', 'Canada');
    """,
    """
    INSERT INTO posts (title, description, user_id)
    VALUES
        ('Feliz', 'Me siento feliz hoy', 1),
        ('Caliente', 'El clima esta caliente hoy', 2),
        ('Ayuda', 'Necesito ayuda en esto', 2),
        ('Buenas noticias', 'Me casare pronto', 1),
        ('Juego interesante', 'Fue genial jugar al tenis', 5),
        ('Fiesta', 'Alguno quiere venir a esta fiesta hoy?', 3);
    """,
    """
    INSERT INTO comments (text, user_id, post_id)
    VALUES
        ('Cuenta conmigo', 1, 6),
        ('Que tipo de ayuda?', 5, 3),
        ('Felicitaciones', 2, 4),
        ('Estuve jugando con Rafael', 4, 5),
        ('Te ayudo con tu tesis?', 2, 3),
        ('Muchas felicitaciones', 5, 4);
    """,
    """
    INSERT INTO likes (user_id, post_id)
    VALUES
        (1, 6), (2, 3), (1, 5), (5, 4), (2, 4), (4, 2), (3, 6);
    """,
]

QUERY_POSTS_WITH_USERS = """
    SELECT users.id, users.name, posts.description
    FROM posts
    INNER JOIN users ON users.id = posts.user_id
"""

QUERY_POSTS_COMMENTS_USERS = """
    SELECT posts.description AS post, comments.text AS comment, users.name
    FROM posts
    INNER JOIN comments ON posts.id = comments.post_id
    INNER JOIN users ON users.id = comments.user_id
"""


def build_demo_database(connection: Connection) -> None:
    """Crea las tablas y carga datos de ejemplo (idempotente)."""
    for statement in CREATE_TABLES:
        execute_query(connection, statement)

    users_count = execute_read_query(connection, "SELECT COUNT(*) FROM users")
    if users_count and users_count[0][0] == 0:
        for statement in SEED_DATA:
            execute_query(connection, statement)


def print_rows(title: str, rows: list[tuple]) -> None:
    print(f"\n-- {title} --")
    for row in rows:
        print(row)


def main() -> None:
    connection = create_connection(DB_PATH)
    if connection is None:
        return

    with connection:
        build_demo_database(connection)

        print_rows("Usuarios", execute_read_query(connection, "SELECT * FROM users"))
        print_rows("Posts", execute_read_query(connection, "SELECT * FROM posts"))
        print_rows(
            "Posts con nombre de usuario",
            execute_read_query(connection, QUERY_POSTS_WITH_USERS),
        )
        print_rows(
            "Posts, comentarios y usuarios",
            execute_read_query(connection, QUERY_POSTS_COMMENTS_USERS),
        )

    connection.close()


if __name__ == "__main__":
    main()