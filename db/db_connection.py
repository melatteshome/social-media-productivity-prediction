#!/usr/bin/env python3
"""
postgres_healthcheck.py  –  tiny helpers to open/close a PostgreSQL session
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv


def connect_db():
    """
    Open a PostgreSQL connection using the DATABASE_URL env-var.

    Returns
    -------
    psycopg2.extensions.connection
        An open connection object on success.
    None
        If the environment variable is missing or the handshake fails.
    """
    load_dotenv()  # inject variables from the local .env file  # :contentReference[oaicite:0]{index=0}
    db_url = os.getenv("DB_URL")

    if not db_url:
        print("❌ DATABASE_URL not found – create a .env file first.")
        return None

    try:
        conn = psycopg2.connect(dsn=db_url)   # :contentReference[oaicite:1]{index=1}
        conn.autocommit = True                # optional: skip implicit txns
        print("✅ Connected to Postgres!")
        return conn
    except psycopg2.Error as err:
        print(f"❌ Could not connect: {err}")
        return None


def close_db(conn):
    """
    Close an existing psycopg2 connection if it is still open.
    """
    if conn and not conn.closed:              # status 0 == OK / open  :contentReference[oaicite:2]{index=2}
        conn.close()
        print("🔒 Connection closed.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    connection = connect_db()                # open session
    try:
        # Do whatever quick health-check or queries you need here
        pass
    finally:
        close_db(connection)                 # clean shutdown  :contentReference[oaicite:3]{index=3}
