import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple, Dict, Any

DB_PATH = os.getenv("EF_DB_PATH", "ef_app.db")


def ensure_directories() -> None:
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/heatmaps", exist_ok=True)


@contextmanager
def get_connection(db_path: str = DB_PATH):
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(db_path: str = DB_PATH) -> None:
    ensure_directories()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                sex TEXT,
                notes TEXT,
                created_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                maternal_rh TEXT,
                fetal_rh TEXT,
                coombs_test TEXT,
                erythroblast_prob REAL,
                risk_score REAL,
                image_path TEXT,
                heatmap_path TEXT,
                created_at TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """
        )


def upsert_patient(patient_id: str, name: str, age: Optional[int], sex: str, notes: str, created_at: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (patient_id, name, age, sex, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                name=excluded.name,
                age=excluded.age,
                sex=excluded.sex,
                notes=excluded.notes
            """,
            (patient_id, name, age, sex, notes, created_at),
        )


def add_prediction(
    patient_id: str,
    maternal_rh: str,
    fetal_rh: str,
    coombs_test: str,
    erythroblast_prob: float,
    risk_score: float,
    image_path: str,
    heatmap_path: str,
    created_at: str,
) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                patient_id, maternal_rh, fetal_rh, coombs_test,
                erythroblast_prob, risk_score, image_path, heatmap_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                maternal_rh,
                fetal_rh,
                coombs_test,
                erythroblast_prob,
                risk_score,
                image_path,
                heatmap_path,
                created_at,
            ),
        )


def list_patients(search: str = "") -> List[Tuple[Any, ...]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        if search:
            like = f"%{search}%"
            cursor.execute(
                """
                SELECT patient_id, name, age, sex, created_at
                FROM patients
                WHERE patient_id LIKE ? OR name LIKE ?
                ORDER BY created_at DESC
                """,
                (like, like),
            )
        else:
            cursor.execute(
                """
                SELECT patient_id, name, age, sex, created_at
                FROM patients
                ORDER BY created_at DESC
                """
            )
        return cursor.fetchall()


def get_patient(patient_id: str) -> Optional[Tuple[Any, ...]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, name, age, sex, notes, created_at
            FROM patients
            WHERE patient_id = ?
            """,
            (patient_id,),
        )
        return cursor.fetchone()


def list_predictions(limit: int = 100) -> List[Tuple[Any, ...]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, risk_score, erythroblast_prob, maternal_rh, fetal_rh, coombs_test, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def list_predictions_for_patient(patient_id: str) -> List[Tuple[Any, ...]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, risk_score, erythroblast_prob, maternal_rh, fetal_rh, coombs_test, image_path, heatmap_path, created_at
            FROM predictions
            WHERE patient_id = ?
            ORDER BY created_at DESC
            """,
            (patient_id,),
        )
        return cursor.fetchall()


def export_predictions_dataframe() -> "pd.DataFrame":
    import pandas as pd  # local import to avoid hard dependency on import order
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT * FROM predictions ORDER BY created_at DESC
            """,
            conn,
        )


def delete_all_data() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM patients")


