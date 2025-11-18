import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=54321,
        dbname="ML",
        user="postgres",
        password="postgres",  # ← ganti sesuai PostgreSQL kamu
        cursor_factory=RealDictCursor
    )
