import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
D = np.array([0,0,1,1], float)

lr = 0.1
epochs = 50

# SGD
w1 = np.zeros(2)
b1 = 0
sgd = []

for e in range(epochs):
    err_sum = 0

    for x,t in zip(X,D):
        y = np.dot(w1,x) + b1
        err = t - y
        w1 += lr * err * x
        b1 += lr * err
        err_sum += err**2

    sgd.append(err_sum/len(X))

# Batch
w2 = np.zeros(2)
b2 = 0
batch = []

for e in range(epochs):
    y = np.dot(X,w2) + b2
    err = D - y

    w2 += lr * np.dot(err,X)
    b2 += lr * np.sum(err)

    batch.append(np.mean(err**2))

print("SGD:", w1, b1)
print("Batch:", w2, b2)

# Graph
plt.plot(sgd, label="SGD")
plt.plot(batch, label="Batch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("SGD vs Batch")
plt.legend()
plt.grid()
plt.show()