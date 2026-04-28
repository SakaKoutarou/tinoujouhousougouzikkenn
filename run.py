import datasets

X,Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)

import regression

model = regression.LinearRegression()
print(model.x)


model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)

print(model.predict(X))

print(model.score(X,Y))