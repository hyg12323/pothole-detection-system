import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "./results/train_exp1_results.csv"

df = pd.read_csv(CSV_PATH)
epochs = df["epoch"]

plt.figure(figsize=(12, 5))

# 1) Box loss
plt.subplot(1, 2, 1)
plt.plot(epochs, df["train/box_loss"], label="Train Box Loss")
plt.plot(epochs, df["val/box_loss"], label="Val Box Loss")
plt.title("Box Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 2) Classification loss
plt.subplot(1, 2, 2)
plt.plot(epochs, df["train/cls_loss"], label="Train Cls Loss")
plt.plot(epochs, df["val/cls_loss"], label="Val Cls Loss")
plt.title("Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
s