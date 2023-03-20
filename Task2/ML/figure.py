import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def readData(fileName):
    df = pd.read_csv(os.path.dirname(
        __file__) + fileName, encoding='unicode_escape')
    return df


df = readData("/train_data.csv")

# print(df)
sns.set_context({"figure.figsize": (8, 8)})
sns.heatmap(df.corr(), annot=True, vmax=1, square=True,
            cmap="Blues", annot_kws={"fontsize": 7})
plt.show()
