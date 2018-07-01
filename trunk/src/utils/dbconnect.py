import os
import pandas as pd
import sqlite3
from sqlite3 import Error

def OpenConnection(db_file='src/sqldb/alzhetect.db'):

    print db_file
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)

        # cursor = conn.cursor()
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # print(cursor.fetchall())
        # print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

def CloseConnection(conn):
    conn.close()

def AddTableCSV(conn, table_name="patients", csvfile='src/test/TADPOLE_test.csv'):
    df = pd.read_csv(csvfile)
    df.to_sql(table_name, conn, if_exists='append', index=False)

def QueryDB(query, conn):
    return pd.read_sql_query(query, conn)

if __name__ == '__main__':
    conn = OpenConnection()
    # AddTableCSV(conn)

    print QueryDB("SELECT RID FROM patients", conn)
    CloseConnection(conn)
