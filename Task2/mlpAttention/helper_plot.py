import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./simple_mlp/simple_mlp_log.csv")

data = data.to_numpy()

train_loss = data[:, 1]
train_acc = data[:, 3]
test_loss = data[:, 2]
test_acc = data[:, 4]

epochs = np.arange(len(train_acc))

plt.title("Train accuracy & Test accuracy")
l1, = plt.plot(epochs, train_loss)
l2, = plt.plot(epochs, test_loss)

plt.legend([l1, l2], ['train', 'test'])
plt.show()