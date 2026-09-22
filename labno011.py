import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
T = np.array([-1,-1,-1,1])

w = np.zeros(2)
b = 0
lr = 1
errors = []

for e in range(20):
    err = 0

    for x,t in zip(X,T):
        y = 1 if np.dot(w,x)+b >= 0 else -1

        if y != t:
            w += lr*t*x
            b += lr*t
            err += 1

    errors.append(err)
    print(e+1, w, b, err)

    if err == 0:
        break

print("Final Weight:", w)
print("Final Bias:", b)

# Error graph
plt.plot(errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Convergence")
plt.grid()
plt.show()

# Decision Boundary
for x,t in zip(X,T):
    plt.scatter(x[0],x[1],
                color='green' if t==1 else 'red',
                marker='o' if t==1 else 'x')

x1 = np.linspace(-2,2,100)
x2 = -(w[0]*x1+b)/w[1]
plt.plot(x1,x2,'b-',label="Decision Boundary")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Bipolar AND")
plt.legend()
plt.grid()
plt.show()