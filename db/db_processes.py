from __future__ import annotations

import os, sys
import psycopg2
from psycopg2 import sql , extras                     
from dotenv import load_dotenv
import pandas as pd


def fetch_all(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT * FROM {tbl}")
               .format(tbl=sql.Identifier(table_name))
        )
        colnames = [desc[0] for desc in cur.description]   
        rows     = cur.fetchall()                          

    return pd.DataFrame(rows, columns=colnames)

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
    print(f"Table '{table_name}' created")


from io import StringIO
from typing import List, Literal, Optional

import pandas as pd
import psycopg2
from psycopg2 import extras


def push_dataframe(
    conn: psycopg2.extensions.connection,
    df: pd.DataFrame,
    table_name: str,
    *,
    columns: Optional[List[str]] = None,
    method: Literal["copy", "values"] = "copy",
) -> int:
    if df.empty:
        print("  DataFrame is empty ")
        return 0

    cols = columns or list(df.columns)
    rows_to_insert = len(df)

    if method == "copy":
        buf = StringIO()
        df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
        buf.seek(0)
        with conn.cursor() as cur:
            cur.copy_from(buf, table_name, sep="\t", null="\\N", columns=cols)
    elif method == "values":
        records = [tuple(x) for x in df.to_numpy()]
        template = "(" + ",".join(["%s"] * len(cols)) + ")"
        query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s"
        with conn.cursor() as cur:
            extras.execute_values(cur, query, records, template=template)
    else:
        raise ValueError("method must be 'copy' or 'values'")

    print(f"🚀 Inserted {rows_to_insert:,} rows into {table_name}")
    return rows_to_insert
