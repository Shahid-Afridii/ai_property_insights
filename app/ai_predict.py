import torch
import numpy as np
import joblib
from app.ai_model import PropertyModel

# ================= LOAD MODEL =================
checkpoint = torch.load("model/model.pt", map_location="cpu")

postcode_enc = joblib.load("model/postcode_enc.pkl")
type_enc = joblib.load("model/type_enc.pkl")
town_enc = joblib.load("model/town_enc.pkl")

model = PropertyModel(
    checkpoint["n_pc"],
    checkpoint["n_typ"],
    checkpoint["n_town"]
)

model.load_state_dict(checkpoint["model"])
model.eval()


# ================= HELPERS =================
def encode_safe(enc, val):
    if val in enc.classes_:
        return enc.transform([val])[0]
    return 0


# ================= MAIN FUNCTION =================
def predict_future(rows):

    if len(rows) < 4:
        return None

    # sort by year
    rows = sorted(rows, key=lambda x: x["year"])

    prices_raw = [r["price"] for r in rows]

    # 🔥 SCALE
    prices = [p / 1_000_000 for p in prices_raw]

    # encode categorical
    pc = encode_safe(postcode_enc, rows[-1]["postcode"])
    typ = encode_safe(type_enc, rows[-1]["type"])
    town = encode_safe(town_enc, rows[-1]["town"])

    pc = torch.tensor([pc])
    typ = torch.tensor([typ])
    town = torch.tensor([town])

    # initial sequence
    seq = prices[-3:].copy()

    def predict_step(seq):
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            return model(x, pc, typ, town).item()

    # ================= RECURSIVE PREDICTION =================
    results = {}
    current_seq = seq.copy()

    for year in range(1, 11):
        next_val = predict_step(current_seq)
        current_seq = current_seq[1:] + [next_val]

        if year in [1, 2, 5, 10]:
            results[year] = next_val * 1_000_000

    base = prices_raw[-1]

    # ================= CALCULATIONS =================
    def pct(a, b):
        return ((b - a) / a) * 100 if a > 0 else 0

    growth_1 = pct(base, results[1])
    growth_5 = pct(base, results[5])
    growth_10 = pct(base, results[10])

    # ================= TREND =================
    if growth_1 > 5:
        trend = "Growing 📈"
    elif growth_1 < -2:
        trend = "Cooling 📉"
    else:
        trend = "Stable ➖"

    # ================= VOLATILITY =================
    diffs = np.diff(prices_raw)
    volatility = np.mean(np.abs(diffs) / prices_raw[:-1]) * 100 if len(diffs) > 0 else 0

    if volatility > 8:
        risk = "High"
    elif volatility > 4:
        risk = "Medium"
    else:
        risk = "Low"

    # ================= INVESTMENT SCORE =================
    score = min(max((growth_5 + growth_10) / 2, 0), 100)

    # ================= CONFIDENCE =================
    match_count = len(rows)

    if match_count > 100:
        confidence = 90
    elif match_count > 50:
        confidence = 80
    elif match_count > 20:
        confidence = 70
    elif match_count > 10:
        confidence = 60
    else:
        confidence = 50

    # ================= VALUATION =================
    fair_value = results[1]

    if base < fair_value * 0.9:
        valuation = "Undervalued 💰"
    elif base > fair_value * 1.1:
        valuation = "Overpriced ⚠️"
    else:
        valuation = "Fair Value"

    # ================= MARKET COMPARISON =================
    avg_growth = (
        np.mean([
            (prices_raw[i] - prices_raw[i - 1]) / prices_raw[i - 1]
            for i in range(1, len(prices_raw))
        ]) * 100
        if len(prices_raw) > 2 else 3
    )

    if growth_5 > avg_growth:
        market_position = "Outperforming Market 🚀"
    elif growth_5 < avg_growth:
        market_position = "Below Market 📉"
    else:
        market_position = "Tracking Market ➖"

    # ================= EXPLANATION =================
    explanation_parts = []

    if trend.startswith("Growing"):
        explanation_parts.append("Strong upward price trend observed")

    if volatility < 4:
        explanation_parts.append("Stable price movement with low volatility")

    if valuation.startswith("Undervalued"):
        explanation_parts.append("Current price is lower than predicted market value")

    if match_count > 30:
        explanation_parts.append("High data confidence due to strong match count")

    explanation = ". ".join(explanation_parts) if explanation_parts else "Moderate market signals"

    # ================= SUMMARY =================
    summary = (
        f"Property shows {trend.lower()} trend with ~{growth_5:.1f}% growth over 5 years. "
        f"Risk level is {risk.lower()} due to {volatility:.1f}% volatility."
    )

    # ================= FINAL OUTPUT =================
    return {
        "price_forecast": {
            "current": round(base, 2),
            "next_year": round(results[1], 2),
            "next_2_year": round(results[2], 2),
            "next_5_year": round(results[5], 2),
            "next_10_year": round(results[10], 2)
        },

        "growth": {
            "1_year_%": round(growth_1, 2),
            "5_year_%": round(growth_5, 2),
            "10_year_%": round(growth_10, 2)
        },

        "insights": {
            "trend": trend,
            "risk": risk,
            "volatility_%": round(volatility, 2),
            "investment_score": round(score, 1),
            "confidence_%": confidence,
            "valuation": valuation,
            "market_position": market_position
        },

        "explanation": explanation,
        "summary": summary
    }