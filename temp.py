import numpy as np
from scipy import signal

X = np.random.randn(5, 5)
kernel = np.random.randn(2, 2)
res = np.zeros((4, 4))

print(X)
# print(np.argmax(X))
# print(np.unravel_index(np.argmax(X), X.shape))
# print(X[np.unravel_index(np.argmax(X), X.shape)])

# print(X)
# print(kernel)
# print(res)
# print("---")

for i in range(4):
    for j in range(4):
        window = X[i:i+2, j:j+2]
        res[i][j] = np.sum(window * kernel)
        # print(window)
        # print("window")
# print("END--------")
# print(res)
# print(np.sum(res))
# print(signal.correlate2d(X, kernel, 'valid'))
# print(np.sum(signal.correlate2d(X, kernel, 'valid')))
# print(res == signal.correlate2d(X, kernel, 'valid'))

# X = [[[1]]]
# Y = [[0]]
# X = np.pad(X, 1)
# Y = np.pad(Y, 1) + 1

# print(Y)
# X[1] += Y
# print(X)
print(np.__version__)
window = np.random.randn(2, 2, 2)
print(window)
ax = np.argmax(window, axis=(1, 2))
print(ax)
