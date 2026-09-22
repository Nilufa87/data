import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target

print("Original Shape:",X.shape)

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

print("Reduced Shape:",X2.shape)
print("Variance:",pca.explained_variance_ratio_)
print("Total Variance:",
      round(sum(pca.explained_variance_ratio_)*100,2),"%")

for i,name in enumerate(iris.target_names):
    plt.scatter(X2[y==i,0],X2[y==i,1],label=name)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.grid()
plt.show()