import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data

sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/Algerian_forest_fires_dataset.csv")

X, y, _ = load_and_preprocess_data(DATA_PATH)

# Chuyển đổi thành DataFrame để dễ trực quan hóa
df = pd.DataFrame(X)
df["Fire_Risk"] = y

df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Histogram of Features", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df["Fire_Risk"], palette="coolwarm", alpha=0.7)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Scatter Plot of Two Features")
plt.legend(title="Fire Risk")
plt.show()
