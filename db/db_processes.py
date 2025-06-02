from __future__ import annotations

import os, sys
import psycopg2
from psycopg2 import sql                          
from dotenv import load_dotenv


def fetch_all(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT * FROM {tbl}")
               .format(tbl=sql.Identifier(table_name))
        )                                           
        rows = cur.fetchall()                      
    return rows


def fetch_by_query(conn, query: str, params: tuple | None = None):

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def drop_table(conn, table_name: str, cascade: bool = True):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {tbl} {}")
               .format(sql.Identifier(table_name),
                       sql.SQL("CASCADE") if cascade else sql.SQL(""))
        )
    print(f" Table '{table_name}' dropped.")


def alter_table(conn, alter_sql: str):
    with conn.cursor() as cur:
        cur.execute(alter_sql)
    print("Table altered.")


def create_table(conn,
                 table_name: str,
                 columns: dict[str, str],
                 if_not_exists: bool = True) -> None:
    if not columns:
        raise ValueError("columns mapping must define at least one column.")

    # Build:  CREATE TABLE [IF NOT EXISTS] <table> ( col1 DDL, col2 DDL, … );
    ddl = sql.SQL("CREATE TABLE {ine} {tbl} ( {cols} )").format(
        ine=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
        tbl=sql.Identifier(table_name),
        cols=sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(defn))
            for col, defn in columns.items()
        )
    )

    with conn.cursor() as cur:
        cur.execute(ddl)
    print(f"📦 Table '{table_name}' created (IF NOT EXISTS = {if_not_exists}).")

