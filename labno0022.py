import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], float)
D = np.array([0,0,1,1], float)

# Settings
alpha = 0.1
np.random.seed(0)
w = np.random.uniform(-0.5, 0.5, 3)
errors = []

# SGD + Delta Rule
for epoch in range(100):
    total_error = 0

    for x, t in zip(X, D):
        net = np.dot(w, x)
        e = t - net
        w += alpha * e * x
        total_error += e**2

    mse = total_error / len(X)
    errors.append(mse)

    if epoch < 10 or (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: W={np.round(w,4)}, MSE={mse:.6f}")

    if mse < 1e-6:
        break

print("\nFinal weights:", w)

# Prediction
print("\nPredictions:")
for x, t in zip(X, D):
    print(x[:2], "Target:", t, "Output:", round(np.dot(w,x),4))

# MSE graph
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("SGD with Delta Rule")
plt.grid()
plt.show()