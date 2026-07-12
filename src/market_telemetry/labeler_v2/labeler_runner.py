from label_generator_v2 import generate_labels
import pandas as pd

df = pd.read_csv("/home/nizami/projects/python/market_telemetry/data/csv/raw_data_2026-06.csv")

df = generate_labels(
    df,
    horizon_rows=90,
    tp_usdt=250
)

counts = df["label"].value_counts().sort_index()

print("\n=== Label statistics ===")
print(f"FLAT : {counts.get(0,0)}")
print(f"FALL : {counts.get(1,0)}")
print(f"RISE : {counts.get(2,0)}")