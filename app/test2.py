from fastapi import FastAPI
from pydantic import BaseModel
import duckdb
import os
from functools import lru_cache

app = FastAPI()

DATA_PATH = "data/"

# ================= INPUT =================
class PropertyInput(BaseModel):
    display_address: str
    post_code: str
    area: str
    building_type: str | None = None


# ================= DB INIT =================
conn = duckdb.connect(database=":memory:")


def load_data():
    print("Loading into DuckDB...")

    files = [
        f"{DATA_PATH}/{f}"
        for f in os.listdir(DATA_PATH)
        if f.endswith(".csv")
    ]

    union_queries = []

    for file in files:
        year = int(os.path.basename(file).split("-")[1].split(".")[0])

        union_queries.append(f"""
            SELECT
                CAST(column01 AS DOUBLE) AS price,
                column03 AS postcode,
                column04 AS type,
                column09 AS street,
                column11 AS town,
                {year} AS year
            FROM read_csv_auto('{file}', header=False)
        """)

    full_query = " UNION ALL ".join(union_queries)

    conn.execute(f"""
        CREATE TABLE properties AS
        SELECT *,
            UPPER(street) AS street_clean,
            REPLACE(UPPER(postcode), ' ', '') AS postcode_clean,
            SUBSTR(REPLACE(UPPER(postcode), ' ', ''), 1, 3) AS outward,
            SUBSTR(REPLACE(UPPER(postcode), ' ', ''), 1, 5) AS micro,
            UPPER(town) AS town_clean
        FROM ({full_query})
    """)

    print("DuckDB Ready ✅")


load_data()


# ================= HELPERS =================
def clean_text(v):
    return str(v).upper().replace(",", "").replace(".", "").strip()


def clean_postcode(pc):
    return str(pc).upper().replace(" ", "")


def rows(cursor):
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ================= QUERY =================
@lru_cache(maxsize=500)
def run_query(street, postcode, area, outward, micro, building_type):
    type_filter = f"AND type = '{building_type}'" if building_type else ""

    # ===== EXACT =====
    exact_q = f"""
        SELECT price, postcode, type, street, town, year
        FROM properties
        WHERE street_clean = '{street}'
        AND postcode_clean = '{postcode}'
        {type_filter}
        LIMIT 200
    """

    exact = rows(conn.execute(exact_q))

    # ===== STREET =====
    street_q = f"""
        SELECT price, postcode, type, street, town, year
        FROM properties
        WHERE street_clean = '{street}'
        AND outward = '{outward}'
        AND postcode_clean != '{postcode}'
        {type_filter}
        LIMIT 200
    """

    street_rows = rows(conn.execute(street_q))

    # ===== 🔥 MAIN LOGIC (UNCHANGED) =====
    if len(exact) > 0 or len(street_rows) > 0:
        return {
            "exactMatches": exact,
            "streetMatches": street_rows,
            "areaMatches": [],
            "counts": {
                "exact": len(exact),
                "street": len(street_rows),
                "area": 0,
                "total": len(exact) + len(street_rows)
            },
            "matchedBy": "Exact + Street"
        }

    # ===== AREA FALLBACK =====
    area_q = f"""
        SELECT price, postcode, type, street, town, year
        FROM properties
        WHERE town_clean = '{area}'
        AND micro = '{micro}'
        AND street_clean != '{street}'
        {type_filter}
        LIMIT 200
    """

    area_rows = rows(conn.execute(area_q))

    return {
        "exactMatches": [],
        "streetMatches": [],
        "areaMatches": area_rows,
        "counts": {
            "exact": 0,
            "street": 0,
            "area": len(area_rows),
            "total": len(area_rows)
        },
        "matchedBy": "Area Match"
    }


# ================= API =================
@app.post("/predict")
def predict_api(data: PropertyInput):

    street = clean_text(data.display_address.split(",")[0])
    postcode = clean_postcode(data.post_code)
    area = clean_text(data.area)
    building_type = data.building_type.upper() if data.building_type else None

    outward = postcode[:3]
    micro = postcode[:5]

    result = run_query(street, postcode, area, outward, micro, building_type)

    return {
        "success": True,
        **result
    }