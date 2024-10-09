from collections.abc import Callable
from pathlib import Path
import os

import numpy as np
import pandas as pd

from neural_network.layers import DenseLayer
from neural_network.network import NeuralNetwork
from neural_network.activation import Tanh, Softmax
from neural_network.loss import MSE
from neural_network.utils.nparray import to_categorical


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=",")
    X = np.array(data.drop(columns=["collision", "start_x", "start_y", "v_x", "v_y"] + [f"2x{i}" for i in range(90)]))
    print(X.shape)
    X = X.reshape((len(X), 180, 1))

    # clean up input data
    X /= 100
    X = np.nan_to_num(X, nan=-1, posinf=-1, neginf=-1)

    Y = np.array(data["collision"])
    Y = to_categorical(Y)
    Y = Y.reshape(len(Y), 2, 1)
    print(np.shape(Y))
    print(Y[0])
    return X, Y


def valid(pred, out):
    return np.argmax(pred) == np.argmax(out)


def analyse_training(
    network: NeuralNetwork,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    valid_fun: Callable,
    epoches: int,
    lr: float,
    model_path: Path,
    model_name: str,
    report_path: Path,
    test_interval: int = 10,
    trainings: int = 3,
):

    for training_num in range(trainings):
        training_report = {
            "training num": training_num,
            "model name": model_name,
            "epoches": epoches,
            "learning_rate": lr,
            "test interval": test_interval,
        }
        network.random_init()
        for epoch_num in range(epoches):
            network.train(X_train, Y_train, 1, lr)

            if epoch_num % test_interval == 0:
                correct = 0

                for inp, out in zip(X_test, Y_test):
                    pred = network.forward(inp)

                    correct += valid_fun(pred, out)

                training_report.update(
                    {
                        f"epoch {epoch_num} correct ": round(
                            correct / len(Y_test) * 100, 2
                        )
                    }
                )
                print(f"Epoche {epoch_num} {round((correct / len(Y_test)) * 100, 2)}%")

        df = pd.DataFrame([training_report])
        df.to_csv(
            report_path, mode="a", index=False, header=not os.path.exists(report_path)
        )

        network.save(str(model_path / f"{model_name}_{epoches}_{training_num}.hdf5"))


def gen_network(layers_n: list):
    network = NeuralNetwork(MSE())

    for i, j in zip(layers_n[:-1], layers_n[1:]):
        network.add_layer(DenseLayer(i, j))
        network.add_layer(Tanh())

    network.add_layer(DenseLayer(layers_n[-1], 2))
    network.add_layer(Softmax())
    return network


LRS = [
        0.1,
        0.05,
        0.01,
        ]

NETWORKS = [
        [180, 90, 30],
        [180, 360, 90, 30],
        [180, 30]
]

XTRAIN, YTRAIN = load_csv(Path("./data/sim-train.csv"))

XTEST, YTEST = load_csv(Path("./data/sim-test.csv"))

XTEST = XTEST[:100]
YTEST = YTEST[:100]

for lr in LRS:
    print(lr)
    for network_structure in NETWORKS:
        network = gen_network(network_structure)

        name = f"collision-{lr*100}e-2-{'-'.join(list(map(str, network_structure)))}-2-tanh-softmax"
        Path(f"{name}_models").mkdir(exist_ok=True)
        Path("training_reports").mkdir(exist_ok=True)

        print(f"Trainig of: {name}")

        analyse_training(
            network,
            XTRAIN,
            YTRAIN,
            XTEST,
            YTEST,
            valid,
            10,
            lr,
            Path(f"./{name}_models"),
            f"{name}",
            Path(f"./training_reports/{name}_training_report.csv"),
            test_interval=1,
        )
