import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Algeria_dataset_clean.csv")

bin_settings = {
    "temperature": {
        "bins": [-1, 25, 30, 35, data["temperature"].max()],
        "labels": ["<25°C", "25–30°C", "30–35°C", ">35°C"]
    },
    "rh": {
        "bins": [0, 30, 50, 70, 100],
        "labels": ["<30%", "30–50%", "50–70%", ">70%"]
    },
    "ws": {
        "bins": [0, 10, 20, data["ws"].max()],
        "labels": ["0–10", "10–20", ">20"]
    },
    "rain": {
        "bins": [-0.1, 0.1, 5, data["rain"].max()],
        "labels": ["0 mm", "0–5 mm", ">5 mm"]
    },
    "ffmc": {
        "bins": [0, 60, 75, 85, data["ffmc"].max()],
        "labels": ["<60", "60–75", "75–85", ">85"]
    },
    "dmc": {
        "bins": [0, 10, 25, 50, data["dmc"].max()],
        "labels": ["<10", "10–25", "25–50", ">50"]
    },
    "dc": {
        "bins": [0, 50, 100, 150, data["dc"].max()],
        "labels": ["<50", "50–100", "100–150", ">150"]
    },
    "isi": {
        "bins": [0, 2, 5, 10, data["isi"].max()],
        "labels": ["<2", "2–5", "5–10", ">10"]
    },
    "bui": {
        "bins": [0, 20, 40, 60, data["bui"].max()],
        "labels": ["<20", "20–40", "40–60", ">60"]
    }
}

for feature, settings in bin_settings.items():

    data[f"{feature}_range"] = pd.cut(
        data[feature],
        bins=settings["bins"],
        labels=settings["labels"],
        include_lowest=True
    )

    avg_fwi = data.groupby(f"{feature}_range", observed=False)["fwi"].mean()

    plt.figure(figsize=(8, 5))
    avg_fwi.plot(kind="bar", edgecolor="black", color="orange")
    plt.title(f"Average FWI vs {feature.upper()} Ranges")
    plt.xlabel(f"{feature.upper()} Range")
    plt.ylabel("Average FWI")

    plt.xticks(rotation=0, ha="center")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
