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

def AddTableCSV(conn, table_name="patients", csvfile='src/sqldb/alzhetect.db'):
    df = pd.read_csv(csvfile)
    df.to_sql(table_name, conn, if_exists='replace', index=False)   # change 'if_exist' to append if necc.

def QueryDB(query, conn):
    return pd.read_sql_query(query, conn)

if __name__ == '__main__':
    conn = OpenConnection("../sqldb/alzhetect.db")

    # AddTableCSV(conn,table_name="patients", csvfile="../train/TADPOLE_train.csv")
    # AddTableCSV(conn,table_name="patients", csvfile="../test/TADPOLE_test.csv")  # change value inside function to append
    # AddTableCSV(conn,table_name="patient_dict", csvfile="../tadpole/TADPOLE_D1_D2_Dict.csv")
    # AddTableCSV(conn,table_name="selected_features", csvfile="../../../docs/selected_features.csv")

    print QueryDB("SELECT * FROM selected_features;", conn)
    print QueryDB("SELECT name FROM sqlite_master WHERE type='table';", conn)

    CloseConnection(conn)
