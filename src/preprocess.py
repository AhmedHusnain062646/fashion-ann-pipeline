import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

data = np.load("data/raw/fashion_mnist.npz")


def per_image_minmax(a):
    # main's approach: scale each image to [0, 1] using its own min/max
    a = a.astype("float32")
    lo = a.min(axis=(1, 2), keepdims=True)
    hi = a.max(axis=(1, 2), keepdims=True)
    return (a - lo) / (hi - lo + 1e-7)


def standardize(train, test):
    # teammate's approach: zero mean / unit variance using train statistics
    train = train.astype("float32")
    test = test.astype("float32")
    mean, std = train.mean(), train.std()
    return (train - mean) / std, (test - mean) / std


method = params["normalization"]
if method == "per_image_minmax":
    x_train_full = per_image_minmax(data["x_train"])
    x_test = per_image_minmax(data["x_test"])
elif method == "standardize":
    x_train_full, x_test = standardize(data["x_train"], data["x_test"])
else:
    raise ValueError(f"Unknown normalization: {method}")

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

print("Normalization:", method)
print("Saved processed data to data/processed/data.npz")
print("train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)
print("pixel range:", float(x_train.min()), "to", float(x_train.max()))