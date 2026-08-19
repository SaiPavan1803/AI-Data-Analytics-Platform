"""Generate a synthetic sales dataset for testing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

start = pd.Timestamp("2023-01-01")
dates = start + pd.to_timedelta(rng.integers(0, 730, N), unit="D")

regions = rng.choice(["North", "South", "East", "West", "Central"],
                     N, p=[0.25, 0.2, 0.2, 0.2, 0.15])
categories = rng.choice(
    ["Electronics", "Apparel", "Home", "Books", "Sports", "Beauty"], N)
channels = rng.choice(["Online", "Retail", "Wholesale"], N, p=[0.55, 0.35, 0.10])
products = rng.choice([f"P-{i:03d}" for i in range(1, 41)], N)

units = rng.integers(1, 12, N)
unit_price = np.round(rng.uniform(10, 800, N), 2)
discount = np.round(rng.uniform(0, 0.35, N), 2)
revenue = np.round(units * unit_price * (1 - discount), 2)
profit = np.round(revenue * rng.uniform(0.05, 0.4, N), 2)

# inject a few anomalies
anom_idx = rng.choice(N, 12, replace=False)
revenue[anom_idx] *= rng.uniform(4, 8, len(anom_idx))
profit[anom_idx] *= rng.uniform(3, 6, len(anom_idx))

df = pd.DataFrame({
    "OrderDate": dates,
    "Region": regions,
    "Category": categories,
    "Channel": channels,
    "Product": products,
    "Units": units,
    "UnitPrice": unit_price,
    "Discount": discount,
    "Revenue": revenue,
    "Profit": profit,
    "CustomerSatisfaction": np.round(rng.uniform(2.5, 5.0, N), 2),
}).sort_values("OrderDate").reset_index(drop=True)

# inject some missing values
miss_idx = rng.choice(N, 30, replace=False)
df.loc[miss_idx, "CustomerSatisfaction"] = np.nan

df.to_csv("output/project/data/sample_sales.csv", index=False)
print("Wrote", len(df), "rows ->", "output/project/data/sample_sales.csv")
