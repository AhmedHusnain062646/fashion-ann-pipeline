import os, yaml, numpy as np
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion_mnist.npz")
x_train = d["x_train"].astype("float32") / 255.0
x_test = d["x_test"].astype("float32") / 255.0
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, d["y_train"], test_size=p["test_size"], random_state=p["seed"])

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/data.npz", x_train=x_tr, y_train=y_tr,
                    x_val=x_val, y_val=y_val, x_test=x_test, y_test=d["y_test"])