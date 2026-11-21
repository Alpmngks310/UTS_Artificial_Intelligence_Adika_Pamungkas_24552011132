"""Exploratory Data Analysis untuk UTS AI Kasus 6. Tidak melakukan training."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fitur import extract_brand

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "kasus06_cars.csv"
OUT = ROOT / "output"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["Brand"] = df["Name"].map(extract_brand)
df["Model_Year"] = pd.to_datetime(df["Year"], errors="coerce").dt.year

# Data dictionary
rows = [
    ("Name", "object", "Nama lengkap kendaraan; diekstraksi menjadi Brand"),
    ("Miles_per_Gallon", "float", "Konsumsi bahan bakar; 8 missing values"),
    ("Cylinders", "integer", "Jumlah silinder mesin"),
    ("Displacement", "float", "Volume/displacement mesin"),
    ("Horsepower", "float", "Tenaga mesin; 6 missing values"),
    ("Weight_in_lbs", "integer", "Bobot kendaraan dalam pon"),
    ("Acceleration", "float", "Ukuran akselerasi pada dataset"),
    ("Year", "date string", "Tanggal model; diekstraksi menjadi Model_Year"),
    ("Origin", "categorical target", "USA, Europe, Japan"),
]
pd.DataFrame(rows, columns=["column", "original_type", "description_or_decision"]).to_csv(OUT / "data_dictionary.csv", index=False)

# Target distribution
fig, ax = plt.subplots(figsize=(7, 4))
df["Origin"].value_counts().reindex(["USA", "Europe", "Japan"]).plot(kind="bar", ax=ax)
ax.set(title="Distribusi Origin", xlabel="Origin", ylabel="Jumlah")
plt.tight_layout(); plt.savefig(FIG / "target_distribution.png", dpi=160); plt.close()

# Numeric distributions
num = ["Miles_per_Gallon", "Cylinders", "Displacement", "Horsepower", "Weight_in_lbs", "Acceleration"]
axes = df[num].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribusi Fitur Numerik")
plt.tight_layout(); plt.savefig(FIG / "numeric_distributions.png", dpi=160); plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Korelasi Antarfitur Numerik")
plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=160); plt.close()

# Brand frequencies; relationship with target used for modeling is generated from train only in latih.py.
df["Brand"].value_counts().rename_axis("Brand").to_frame("count").to_csv(OUT / "brand_frequency.csv")
print("EDA selesai. Shape:", df.shape)
print("Missing:\n", df.isna().sum().to_string())
print("Target:\n", df.Origin.value_counts().to_string())
print("Normalized brands:", df.Brand.nunique())
print("Tahun model:", sorted(df.Model_Year.dropna().unique().tolist()))
print("Output:", OUT)
