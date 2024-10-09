from pathlib import Path

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_network.layers import DenseLayer
from neural_network.network import NeuralNetwork
from neural_network.activation import Softmax, Tanh
from neural_network.loss import MSE
from neural_network.utils.nparray import to_categorical


def load_csv_180(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=",")
    X = np.array(data.drop(columns=["collision", "start_x", "start_y", "v_x", "v_y"]+[f"2x{i}" for i in range(90)]))
    X = X.reshape((len(X), 180, 1))

    # clean up input data
    X /= 100
    X = np.nan_to_num(X, nan=-1, posinf=-1, neginf=-1)

    Y = np.array(data["collision"])
    Y = to_categorical(Y)
    Y = Y.reshape(len(Y), 2, 1)

    return X, Y


def load_csv_270(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=",")
    X = np.array(data.drop(columns=["collision", "start_x", "start_y", "v_x", "v_y"]))
    X = X.reshape((len(X), 270, 1))

    # clean up input data
    X /= 100
    X = np.nan_to_num(X, nan=-1, posinf=-1, neginf=-1)

    Y = np.array(data["collision"])
    Y = to_categorical(Y)
    Y = Y.reshape(len(Y), 2, 1)

    return X, Y


def load_csv_start_positions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=",")
    return data[["collision", "start_x", "start_y", "v_x", "v_y"]]


def valid(pred, out):
    return np.argmax(pred) == np.argmax(out)


def valid_perc(pred, out, perc=0.7):
    return bool((np.argmax(pred) == np.argmax(out)) and (pred[np.argmax(pred)] >= perc))


def gen_network(layers_n: list):
    network = NeuralNetwork(MSE())

    for i, j in zip(layers_n[:-1], layers_n[1:]):
        network.add_layer(DenseLayer(i, j))
        network.add_layer(Tanh())

    network.add_layer(DenseLayer(layers_n[-1], 2))
    network.add_layer(Softmax())
    return network

PATH = Path("./data/sim-test.csv")
XTEST_270, YTEST_270 = load_csv_270(PATH)
XTEST_180, YTEST_180 = load_csv_180(PATH)

STARTS = load_csv_start_positions(PATH)

MODELS = glob.glob(os.path.join("./best-models/", '*.hdf5'))
print(len(XTEST_180))
for model in MODELS:
    name = model.split("/")[-1][:-5]
    structure = list(map(int, name.split("-")[3:-2]))
    network = gen_network(structure[:-1])

    network.load(model)
    print(name)

    if structure[0] == 180:
        x,y = XTEST_180, YTEST_180

    else:
        x,y = XTEST_270, YTEST_270

    correct = 0
    correct_70 = 0

    for inp, out in zip(x, y):
        pred = network.forward(inp)
        correct += valid(pred, out)
        correct_70 += valid_perc(pred, out, 0.7)

    print(f"{round(correct/len(y)*100, 2)}%")
    print(f"{round(correct_70/len(y)*100, 2)}%")
    print()
"""
correct = 0
correct_70 = 0
data = {"pred": [], "pred_perc": [], "expected": [], "correct": []}

for inp, out in zip(XTEST, YTEST):
    pred = network.forward(inp)
    data["pred"].append(np.argmax(pred))
    data["expected"].append(np.argmax(out))
    data["pred_perc"].append(pred[np.argmax(pred)][0])
    data["correct"].append(valid(pred, out))
    correct += valid(pred, out)
    correct_70 += valid_perc(pred, out)


df = pd.DataFrame(data)
df.to_csv(REPORT_PATH, header=True, index=True, mode="w")


print(f"Correct predictions percentage (>50%): {round(correct/len(YTEST)*100, 2)}%")
print(f"Correct predictions percentage (>70%): {round(correct_70/len(YTEST)*100, 2)}%")
plt.scatter(STARTS["start_x"], STARTS["start_y"], c=STARTS["collision"].map({True: "green", False: "red"}), alpha=0.25)
plt.show()

corr = STARTS.loc[df.index[df["correct"]]]
plt.scatter(corr["start_x"], corr["start_y"], c=corr["collision"].map({True: "green", False: "red"}), alpha=0.25)
plt.show()


wrong = STARTS.loc[df.index[df["correct"] == False]]

norm = np.sqrt(wrong["v_x"]**2 + wrong["v_y"]**2)
norm[norm == 0] = 1
norm_vx = 2 * wrong["v_x"] / norm
norm_vy = 2 * wrong["v_y"] / norm

point_colors = wrong["collision"].map({True: "green", False: "red"})
vec_colors = np.where((wrong["v_x"] < 0) & (wrong["v_y"] < 0), 'black', 'blue')

plt.scatter(wrong["start_x"], wrong["start_y"], c=point_colors)
plt.quiver(wrong["start_x"], wrong["start_y"], norm_vx, norm_vy, angles="xy", scale_units='xy', scale=1, color=vec_colors, alpha=0.4)
plt.show()

"""
