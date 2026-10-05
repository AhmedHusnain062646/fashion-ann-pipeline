import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")
loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = model.predict(d["x_test"]).argmax(axis=1)

os.makedirs("evaluation", exist_ok=True)
ConfusionMatrixDisplay.from_predictions(d["y_test"], pred)
plt.savefig("evaluation/confusion_matrix.png")
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"), indent=2)