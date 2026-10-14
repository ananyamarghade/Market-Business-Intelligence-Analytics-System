import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START_YEAR, START_MONTH, N = 2023, 10, 36

CATEGORIES = {
    "Electronics": (180, 0.009, [.92, .88, .90, .92, .95, .95, .97, 1.00, 1.04, 1.10, 1.38, 1.30]),
    "Apparel": (120, -0.005, [.80, .85, 1.00, 1.10, 1.10, 1.00, .90, 1.05, 1.10, 1.05, 1.12, 1.20]),
    "Home": (90, 0.015, [.90, .90, 1.00, 1.10, 1.15, 1.05, 1.00, 1.00, 1.00, 1.00, 1.05, 1.02]),
    "Grocery": (220, 0.004, [1.00, .95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.15]),
    "Beauty": (60, 0.011, [.95, 1.10, 1.00, 1.00, 1.10, 1.00, .95, .95, 1.00, 1.00, 1.10, 1.20]),
}
REGIONS = {"North": 1.00, "South": 0.82, "East": 1.18, "West": 0.96}


def build():
    rng = np.random.default_rng(11)
    t = np.arange(N)
    m = (START_MONTH - 1 + t) % 12
    promo = 45 + np.where(m >= 9, 12, 0) + (rng.random(N) - .5) * 14
    temp = 25 + 7 * np.sin((m - 2) / 12 * 2 * np.pi) + (rng.random(N) - .5) * 2
    price = 100 + t * .12 + (rng.random(N) - .5) * 5 - (promo - 50) * .08

    rows = []
    for cat, (base, growth, season) in CATEGORIES.items():
        for reg, mult in REGIONS.items():
            noise = 1 + (rng.random(N) - .5) * .08
            rev = (base * mult * (1 + growth) ** t * np.array(season)[m]
                   * (1 + .006 * (promo - 50)) * (1 - .006 * (price - 100)) * noise)
            rows.extend((i, cat, reg, rev[i]) for i in range(N))

    df = pd.DataFrame(rows, columns=["t", "category", "region", "revenue"])
    total = df.groupby("t")["revenue"].sum().to_numpy()
    traffic = total * .85 * (1 + (rng.random(N) - .5) * .12)
    dates = pd.date_range(f"{START_YEAR}-{START_MONTH:02d}-01", periods=N, freq="MS")
    drivers = pd.DataFrame({"t": t, "promo": promo, "price_index": price, "temperature": temp, "traffic": traffic})
    df = df.merge(drivers, on="t")
    df["month"] = dates[df["t"]]
    return df[["month", "category", "region", "revenue", "promo", "price_index", "temperature", "traffic"]].round({"revenue": 4, "promo": 4, "price_index": 4, "temperature": 4, "traffic": 4})


def export_js(df):
    ordered = df.sort_values("month")
    revenue = {c: {r: ordered[(ordered.category == c) & (ordered.region == r)]["revenue"].round(3).tolist()
                   for r in REGIONS} for c in CATEGORIES}
    drv = ordered.drop_duplicates("month")
    payload = {
        "start": [START_YEAR, START_MONTH - 1],
        "months": N,
        "categories": list(CATEGORIES),
        "regions": list(REGIONS),
        "revenue": revenue,
        "promo": drv["promo"].round(3).tolist(),
        "price": drv["price_index"].round(3).tolist(),
        "temp": drv["temperature"].round(3).tolist(),
        "traffic": drv["traffic"].round(3).tolist(),
    }
    (ROOT / "dashboard" / "data.js").write_text("window.SALES = " + json.dumps(payload) + ";\n")


if __name__ == "__main__":
    frame = build()
    frame.to_csv(ROOT / "data" / "monthly_sales.csv", index=False)
    export_js(frame)
    print(f"{len(frame)} rows written")
