import numpy as np
import matplotlib.pyplot as plt

# Digits 1-5
D = np.eye(5)

X = np.array([
[0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
[1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,1],
[1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],
[0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,1,1,1,1, 0,0,0,1,0],
[1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0]
], float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e = np.exp(x-np.max(x))
    return e/e.sum()

np.random.seed(3)
W1 = np.random.randn(10,25)*0.5
W2 = np.random.randn(5,10)*0.5
lr = 0.5

# Training
for epoch in range(3000):
    for x,d in zip(X,D):
        h = sigmoid(W1@x)
        y = softmax(W2@h)
        e = d-y

        dh = h*(1-h)*(W2.T@e)
        W2 += lr*np.outer(e,h)
        W1 += lr*np.outer(dh,x)

# Show digits
plt.figure(figsize=(8,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X[i].reshape(5,5), cmap='gray')
    plt.title(i+1)
    plt.axis('off')
plt.show()

# Test
print("Recognition Result\n")

for i,x in enumerate(X):
    y = softmax(W2 @ sigmoid(W1@x))
    p = np.argmax(y)+1
    c = np.max(y)

    print(f"Actual: {i+1}  Predicted: {p}  Confidence: {c:.2f}")