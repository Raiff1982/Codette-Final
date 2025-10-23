"""
SQLiteManager: Centralized data access layer for SQLite operations.
Used by aegis.py and aegis_integration/aegis.py to abstract raw SQL/database logic.
"""
import sqlite3
from threading import Lock
from typing import Any, Optional, Dict, List

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = Lock()

    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            self.conn.commit()

    def fetchall(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchall()

    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchone()

    def close(self):
        self.conn.close()
