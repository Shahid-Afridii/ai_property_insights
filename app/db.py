import duckdb
import os
import pandas as pd

DB = duckdb.connect(database=':memory:')

DATA_PATH = "data/"

def load_data():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".xlsx")]

    dfs = []
    for file in files:
        year = int(file.split('-')[1].split('.')[0])
        df = pd.read_excel(os.path.join(DATA_PATH, file))
        df['year'] = year
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    full_df.columns = [
        "id","price","date","postcode","type","new","tenure",
        "house_no","extra","street","locality","town","district","county","flag","year"
    ]

    DB.execute("CREATE TABLE properties AS SELECT * FROM full_df")