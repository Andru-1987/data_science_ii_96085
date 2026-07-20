"""
Introduccion a bases de datos documentales con TinyDB
========================================================

TinyDB es a las bases documentales lo que sqlite3 es a las relacionales:
una libreria liviana, sin servidor, que guarda todo en un unico archivo
(en este caso JSON) y se instala con `pip install tinydb`.


1. Queries expresivas con `Query()` en vez de escribir el `for` a mano
   (`Post.likes` funciona parecido a `posts.likes` en Mongo).
2. `insert` / `insert_multiple` que generan el id (`doc_id`) automaticamente.
3. `update`, `remove`, `upsert` para modificar documentos existentes.
4. Tablas (`db.table(...)`) como equivalente directo a "colecciones", cada
   una con su propio archivo logico dentro del mismo JSON.

El modelo de datos sigue siendo el mismo (blog con users/posts), y los
comentarios/likes se mantienen EMBEBIDOS dentro del documento del post,
que es el patron tipico en bases documentales (ver el ejemplo anterior con
JSON puro para la explicacion conceptual de embeber vs. referenciar).
"""

from __future__ import annotations

from pathlib import Path

from tinydb import Query, TinyDB

DB_PATH = Path(__file__).parent / "blog_tinydb.json"


def create_connection(path: Path) -> TinyDB:
    """Abre (o crea) la base TinyDB en `path`."""
    db = TinyDB(path, ensure_ascii=False, encoding="utf-8")
    print(f"Conexion a TinyDB exitosa ({path})")
    return db


def build_demo_database(db: TinyDB) -> None:
    """Carga datos de ejemplo (idempotente: no duplica si ya hay usuarios)."""
    users = db.table("users")
    posts = db.table("posts")

    if users:
        return

    users.insert_multiple(
        [
            {"name": "James", "age": 25, "gender": "hombre", "nationality": "USA"},
            {"name": "Leila", "age": 32, "gender": "mujer", "nationality": "France"},
            {"name": "Brigitte", "age": 35, "gender": "mujer", "nationality": "England"},
            {"name": "Mike", "age": 40, "gender": "hombre", "nationality": "Denmark"},
            {"name": "Elizabeth", "age": 21, "gender": "mujer", "nationality": "Canada"},
        ]
    )

    posts.insert_multiple(
        [
            {
                "title": "Feliz",
                "description": "Me siento feliz hoy",
                "user_id": 1,
                "comments": [{"text": "Cuenta conmigo", "user_id": 1}],
                "likes": [1, 2],
            },
            {
                "title": "Caliente",
                "description": "El clima esta caliente hoy",
                "user_id": 2,
                "comments": [],
                "likes": [4],
            },
            {
                "title": "Ayuda",
                "description": "Necesito ayuda en esto",
                "user_id": 2,
                "comments": [
                    {"text": "Que tipo de ayuda?", "user_id": 5},
                    {"text": "Te ayudo con tu tesis?", "user_id": 2},
                ],
                "likes": [2],
            },
            {
                "title": "Buenas noticias",
                "description": "Me casare pronto",
                "user_id": 1,
                "comments": [
                    {"text": "Felicitaciones", "user_id": 2},
                    {"text": "Muchas felicitaciones", "user_id": 5},
                ],
                "likes": [5, 2],
            },
            {
                "title": "Juego interesante",
                "description": "Fue genial jugar al tenis",
                "user_id": 5,
                "comments": [{"text": "Estuve jugando con Rafael", "user_id": 4}],
                "likes": [1],
            },
            {
                "title": "Fiesta",
                "description": "Alguno quiere venir a esta fiesta hoy?",
                "user_id": 3,
                "comments": [],
                "likes": [1, 3],
            },
        ]
    )


def print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n-- {title} --")
    for row in rows:
        print(row)


def posts_with_user_names(db: TinyDB) -> list[dict]:
    """Equivalente al JOIN posts-users, resuelto con dos queries de TinyDB."""
    users_by_id = {user.doc_id: user["name"] for user in db.table("users")}
    return [
        {"user": users_by_id.get(post["user_id"]), "description": post["description"]}
        for post in db.table("posts")
    ]


def main() -> None:
    db = create_connection(DB_PATH)
    build_demo_database(db)

    Post = Query()

    print_rows("Usuarios", db.table("users").all())
    print_rows("Posts", db.table("posts").all())
    print_rows("Posts con nombre de usuario", posts_with_user_names(db))
    print_rows(
        "Posts con mas de un like",
        db.table("posts").search(Post.likes.test(lambda likes: len(likes) > 1)),
    )
    print_rows(
        "Posts de Leila (user_id 2)",
        db.table("posts").search(Post.user_id == 2),
    )

    db.close()


if __name__ == "__main__":
    main()