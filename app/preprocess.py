import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = "data/"
MODEL_DIR = "model"
SCALE = 1_000_000.0

postcode_enc = LabelEncoder()
type_enc = LabelEncoder()
town_enc = LabelEncoder()


def normalize(s):
    return str(s).upper().strip()


def load_all():
    dfs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            print("Loading:", file)

            year = int(file.split('-')[1].split('.')[0])

            df = pd.read_csv(
                os.path.join(DATA_PATH, file),
                header=None,
                dtype=str,
                low_memory=False
            )

            # HM Land Registry columns
            df = df[[1, 3, 4, 9, 11]]

            df.columns = ["price", "postcode", "type", "street", "town"]

            # normalize
            df["postcode"] = df["postcode"].apply(normalize)
            df["type"] = df["type"].apply(normalize)
            df["street"] = df["street"].apply(normalize)
            df["town"] = df["town"].apply(normalize)

            df["year"] = year

            dfs.append(df)

    if not dfs:
        raise Exception("No CSV files found")

    return pd.concat(dfs, ignore_index=True)


def prepare(df):
    df = df.dropna()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # scale
    df["price"] = df["price"] / SCALE

    # encode
    df["postcode_enc"] = postcode_enc.fit_transform(df["postcode"])
    df["type_enc"] = type_enc.fit_transform(df["type"])
    df["town_enc"] = town_enc.fit_transform(df["town"])

    # save encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(postcode_enc, f"{MODEL_DIR}/postcode_enc.pkl")
    joblib.dump(type_enc, f"{MODEL_DIR}/type_enc.pkl")
    joblib.dump(town_enc, f"{MODEL_DIR}/town_enc.pkl")
    joblib.dump({"scale": SCALE}, f"{MODEL_DIR}/meta.pkl")

    df = df.sort_values(["postcode", "street", "year"])

    sequences, targets = [], []
    pc_list, type_list, town_list = [], [], []

    for (pc, st), g in df.groupby(["postcode", "street"]):
        prices = g["price"].values

        if len(prices) < 4:
            continue

        pc_val = g["postcode_enc"].iloc[0]
        type_val = g["type_enc"].iloc[0]
        town_val = g["town_enc"].iloc[0]

        for i in range(len(prices) - 3):
            sequences.append(prices[i:i+3])
            targets.append(prices[i+3])

            pc_list.append(pc_val)
            type_list.append(type_val)
            town_list.append(town_val)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(pc_list, dtype=np.int64),
        np.array(type_list, dtype=np.int64),
        np.array(town_list, dtype=np.int64),
        len(postcode_enc.classes_),
        len(type_enc.classes_),
        len(town_enc.classes_)
    )