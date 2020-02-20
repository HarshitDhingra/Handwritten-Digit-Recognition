import numpy as np
import pandas as pd


def Euclidean_distance(x1, x2):
    return np.sum((x1 - x2) ** 2) ** .5


def Knn(X, Y, test, k=5):
    n = X.shape[0]
    h = []
    for i in range(n):
        s = Euclidean_distance(test, X[i])
        h.append((s, Y[i]))
    # print(h[7])
    # print(Y[7])
    h = sorted(h)
    h = np.array(h[:k])
    z = np.unique(h[:, 1], return_counts=True)
    index = np.argmax(z[1])
    prediction = z[0][index]
    return int(prediction)


df = pd.read_csv("mnist/train.csv")
#print(df.head())
#print(df.shape)
data = df.values

X = data[:41500, 1:]
Y = data[:41500, 0]
x = data[41500:, 1:]
y = data[41500:, 0]
#print(x.shape)
# print(X.shape)
test = x[71]
# print(X[3])
pred = Knn(X, Y, test)
print(pred)
print(y[71])


def accuracy(X, Y, x, y):
    n = x.shape[0]
    count = 0
    totalcount = 0
    for i in range(n):
        pred = Knn(X, Y, x[i])
        if pred == y[i]:
            count += 1
            # print(count)
        totalcount += 1
    print(count)
    print(totalcount)
    return (count/totalcount)


ans = accuracy(X, Y, x, y)
print(ans*100)

# count = 493
# totalcount = 500

#  Accuracy = 98.6 %