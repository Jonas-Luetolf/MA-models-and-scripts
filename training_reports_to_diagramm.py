import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

PATH = "./models/training_reports/"

csv_files = glob.glob(os.path.join(PATH, "*.csv"))

dataframes = []

for file in csv_files:
    filename = os.path.basename(file)
    match = re.search(r"collision-(\d+(\.\d+)?e-[\d]+)", filename)

    if match:
        learning_rate = match.group(1)
        df = pd.read_csv(file)
        df["learning_rate"] = learning_rate
        dataframes.append(df)

data = pd.concat(dataframes)

learning_rates = data["learning_rate"].unique()
print(learning_rates)

for lr in learning_rates:
    print(float(lr))
    lr_data = data[data["learning_rate"] == lr].drop(columns=["learning_rate"])
    print(len(lr_data) / 3)

    avg_data = lr_data.groupby("model name").mean().reset_index()

    plt.figure(figsize=(12, 6))

    for model_name in avg_data["model name"].unique():
        model_data = avg_data[avg_data["model name"] == model_name]
        plt.plot(range(10), model_data.iloc[0, 4:], marker="o", label=model_name)

    plt.xlabel("Epoche")
    plt.ylabel("Durchschnittliche Korrektheit (%)")
    plt.title(f"Durchschnittliche Korrektheit über Epoche für Lernrate {float(lr)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"training-{lr}.png")
    plt.savefig(f"training-{lr}.png")
    plt.close()

print("Diagramme wurden erstellt und gespeichert.")
