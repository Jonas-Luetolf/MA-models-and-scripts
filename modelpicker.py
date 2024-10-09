import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import shutil
from pathlib import Path

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
print(dataframes[0].keys())

best = []
for data in dataframes:
    file = f"{data["model name"][0]}_10_{np.argmax(data["epoch 9 correct "])}.hdf5"
    name = data["model name"][0]
    best.append((name, file))

for model in best:
    shutil.copy(Path(f"./models/{model[0]}_models/{model[1]}"), Path(f"./best-models/{model[0]}.hdf5"))
