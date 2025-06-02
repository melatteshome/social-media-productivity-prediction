#!/usr/bin/env python3
"""
Simple helpers for opening a PostgreSQL connection.

Reads DATABASE_URL from .env or the shell.
Works with both psycopg2 and (optionally) SQLAlchemy.
"""

import os
from contextlib import contextmanager
from dotenv import load_dotenv          
import psycopg2                         

load_dotenv()                           

# Fallback URL has the same shape you showed earlier
DATABASE_URL = os.getenv("DB_URL",)

# ── 2. psycopg2 connection helper ───────────────────────────────────────────
@contextmanager
def pg_connection():
    """
    Yields a psycopg2 connection inside a context-manager.

    Usage:
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT now();")
                print(cur.fetchone())
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn                 # commits automatically on normal exit
        conn.commit()
        print('connected')
    except Exception:              # rolls back on error
        conn.rollback()
        raise
    finally:
        conn.close()               # always close – even if exceptions occur
