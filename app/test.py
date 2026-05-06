from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import joblib

app = FastAPI()

DATA_PATH = "data/"

# ================= INPUT =================
class PropertyInput(BaseModel):
    display_address: str
    post_code: str
    area: str
    building_type: str | None = None


# ================= LOAD ENCODER =================
type_enc = joblib.load("model/type_enc.pkl")
ALL_TYPES = type_enc.classes_.tolist()


# ================= LOAD DATA =================
def load_all_data():
    dfs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            print("Loading:", file)

            year = int(file.split('-')[1].split('.')[0])

            df = pd.read_csv(
                os.path.join(DATA_PATH, file),
                header=None,
                usecols=[1, 3, 4, 9, 11]
            )

            df.columns = ["price", "postcode", "type", "street", "town"]
            df["year"] = year

            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print("Preprocessing...")

    df["street_clean"] = df["street"].astype(str).str.upper()
    df["postcode_clean"] = df["postcode"].astype(str).str.replace(" ", "").str.upper()
    df["outward"] = df["postcode_clean"].str[:3]
    df["micro"] = df["postcode_clean"].str[:5]
    df["town_clean"] = df["town"].astype(str).str.upper()

    print("Data Ready ✅")

    return df


df_global = load_all_data()


# ================= HELPERS =================
def clean_text(v):
    return str(v).upper().replace(",", "").replace(".", "").strip()


def clean_postcode(pc):
    return str(pc).upper().replace(" ", "")


def format_rows(df):
    return df[["price", "postcode", "type", "street", "town", "year"]].to_dict("records")


# ================= MATCHING =================
def match_rows(data: PropertyInput):
    street = clean_text(data.display_address.split(",")[0])
    postcode = clean_postcode(data.post_code)
    area = clean_text(data.area)
    building_type = data.building_type.upper() if data.building_type else None

    outward = postcode[:3]
    micro = postcode[:5]

    df = df_global

    # ===== EXACT =====
    exact = df[
        (df["street_clean"] == street) &
        (df["postcode_clean"] == postcode)
    ]

    # ===== STREET =====
    street_match = df[
        (df["street_clean"] == street) &
        (df["outward"] == outward) &
        (df["postcode_clean"] != postcode)
    ]

    # ===== AREA =====
    area_match = df[
        (df["town_clean"] == area) &
        (df["micro"] == micro) &
        (df["street_clean"] != street)
    ]

    # ===== TYPE FILTER =====
    if building_type and building_type in ALL_TYPES:
        exact = exact[exact["type"] == building_type]
        street_match = street_match[street_match["type"] == building_type]
        area_match = area_match[area_match["type"] == building_type]

    # ===== 🔥 FINAL LOGIC =====
    if len(exact) > 0 or len(street_match) > 0:
        return {
            "exactMatches": format_rows(exact.head(200)),
            "streetMatches": format_rows(street_match.head(200)),
            "areaMatches": [],

            "counts": {
                "exact": len(exact),
                "street": len(street_match),
                "area": 0,
                "total": len(exact) + len(street_match)
            },

            "matchedBy": "Exact + Street"
        }

    else:
        return {
            "exactMatches": [],
            "streetMatches": [],
            "areaMatches": format_rows(area_match.head(200)),

            "counts": {
                "exact": 0,
                "street": 0,
                "area": len(area_match),
                "total": len(area_match)
            },

            "matchedBy": "Area Match"
        }


# ================= API =================
@app.post("/predict")
def predict_api(data: PropertyInput):

    result = match_rows(data)

    return {
        "success": True,
        **result
    }