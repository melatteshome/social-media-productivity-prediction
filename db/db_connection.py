
import os
import sys
import psycopg2
from dotenv import load_dotenv


def connect_db():

    load_dotenv() 
    db_url = os.getenv("DB_URL")

    if not db_url:
        print(" DATABASE_URL not found – create a .env file first.")
        return None

    try:
        conn = psycopg2.connect(dsn=db_url)  
        conn.autocommit = True                
        print("Connected to Postgres!")
        return conn
    except psycopg2.Error as err:
        print(f" Could not connect: {err}")
        return None


def close_db(conn):
    """
    Close an existing psycopg2 connection if it is still open.
    """
    if conn and not conn.closed:              
        conn.close()
        print(" Connection closed.")

