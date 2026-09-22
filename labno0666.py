import numpy as np
import matplotlib.pyplot as plt

x1, x2 = 0.05, 0.10
t1, t2 = 0.01, 0.99

w1,w2,w3,w4 = .15,.20,.25,.30
w5,w6,w7,w8 = .40,.45,.50,.55
b1,b2 = .35,.60

lr = .5
losses = []

def sig(x):
    return 1/(1+np.exp(-x))

for epoch in range(10000):

    # Forward
    h1 = sig(w1*x1 + w3*x2 + b1)
    h2 = sig(w2*x1 + w4*x2 + b1)

    y1 = sig(w5*h1 + w7*h2 + b2)
    y2 = sig(w6*h1 + w8*h2 + b2)

    # Error
    loss = .5*((t1-y1)**2 + (t2-y2)**2)
    losses.append(loss)

    # Backpropagation
    d1 = (y1-t1)*y1*(1-y1)
    d2 = (y2-t2)*y2*(1-y2)

    dh1 = (d1*w5+d2*w6)*h1*(1-h1)
    dh2 = (d1*w7+d2*w8)*h2*(1-h2)

    # Update weights
    w1 -= lr*dh1*x1
    w2 -= lr*dh2*x1
    w3 -= lr*dh1*x2
    w4 -= lr*dh2*x2

    w5 -= lr*d1*h1
    w6 -= lr*d2*h1
    w7 -= lr*d1*h2
    w8 -= lr*d2*h2

    # Update bias
    b1 -= lr*(dh1+dh2)
    b2 -= lr*(d1+d2)

print("Final Output:")
print("y1 =", round(y1,4), " Target =", t1)
print("y2 =", round(y2,4), " Target =", t2)
print("Loss =", round(loss,6))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()