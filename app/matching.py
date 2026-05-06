from .db import DB

def clean_text(v):
    return str(v).upper().strip()

def normalize_postcode(pc):
    return str(pc).replace(" ", "").upper()

def read_relevant_rows(street, postcode, area):
    street = clean_text(street)
    postcode = normalize_postcode(postcode)
    outward = postcode[:3]
    area = clean_text(area)

    query = f"""
    WITH base AS (
        SELECT
            regexp_replace(upper(street), '[^A-Z0-9 ]', '', 'g') AS clean_street,
            replace(upper(postcode), ' ', '') AS clean_pc,
            split_part(upper(postcode), ' ', 1) AS outward,
            upper(town) AS clean_area,
            price, year, type, postcode, street, town
        FROM properties
        WHERE price BETWEEN 30000 AND 2000000
    )
    SELECT *
    FROM base
    WHERE
        (clean_street = '{street}' AND clean_pc = '{postcode}')
        OR
        (clean_street = '{street}' AND outward = '{outward}')
        OR
        (clean_area = '{area}' AND outward = '{outward}')
    """

    return DB.execute(query).fetchall()