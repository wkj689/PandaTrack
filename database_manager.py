import sqlite3
import pandas as pd
from pathlib import Path


class PandaDatabaseManager:
    def __init__(self, database_path: str):
        """
        database_path: Previously this argument was an .xlsx path
        Here it is automatically replaced with .sqlite
        """
        self.database_path = Path(database_path).with_suffix(".sqlite")
        self.table_name = "analysis_results"
        self._init_database()

    def _get_connection(self):
        return sqlite3.connect(self.database_path)

    def _init_database(self):
        with self._get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """)

    def _get_existing_columns(self, conn):
        cursor = conn.execute(f"PRAGMA table_info({self.table_name})")
        return {row[1] for row in cursor.fetchall()}

    def _add_missing_columns(self, conn, new_columns):
        for col in new_columns:
            conn.execute(
                f'ALTER TABLE {self.table_name} ADD COLUMN "{col}" TEXT'
            )

    def add_analysis_result(self, result: dict):
        if not isinstance(result, dict):
            raise ValueError("result must be a dict")

        excluded_columns = {
            "absent_time",
            "moving_time",
            "avg_speed",
            "max_speed",
            "static_time"
        }

        result = {
            key: value
            for key, value in result.items()
            if key not in excluded_columns
        }

        with self._get_connection() as conn:
            existing_cols = self._get_existing_columns(conn)

            new_cols = set(result.keys()) - existing_cols
            if new_cols:
                self._add_missing_columns(conn, new_cols)

            columns = ", ".join(f'"{k}"' for k in result.keys())
            placeholders = ", ".join("?" for _ in result)
            values = [str(v) if v is not None else None for v in result.values()]

            conn.execute(
                f"""
                INSERT INTO {self.table_name} ({columns})
                VALUES ({placeholders})
                """,
                values
            )

    def load_database(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql(
                f"SELECT * FROM {self.table_name} ORDER BY id",
                conn
            )
