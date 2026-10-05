import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

data = np.load("data/raw/fashion_mnist.npz")

# --- normalization ---
# main: per-image min-max scaling to [0, 1]
def per_image_minmax(a):
    a = a.astype("float32")
    lo = a.min(axis=(1, 2), keepdims=True)
    hi = a.max(axis=(1, 2), keepdims=True)
    return (a - lo) / (hi - lo + 1e-7)


x_train_full = per_image_minmax(data["x_train"])
x_test = per_image_minmax(data["x_test"])
# --- end normalization ---

y_train_full = data["y_train"]
y_test = data["y_test"]

# Split a validation set out of the training data
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=params["test_size"],
    random_state=params["seed"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
)

print("Saved processed data to data/processed/data.npz")
print("train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)
print("pixel range:", float(x_train.min()), "to", float(x_train.max()))