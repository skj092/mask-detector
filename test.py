from sklearn.datasets import fetch_openml
import time 

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

tic = time.time()
X = X.values
tok = time.time()
print('total time taken to load X', toc-tic)
y = y.astype(int).values 

print(X.shape)
print(y.shape)