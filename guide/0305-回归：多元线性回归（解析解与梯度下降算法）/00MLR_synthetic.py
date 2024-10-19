import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([2,5,8]).reshape((-1, 1))
y = np.array([4,1,9])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)

print('intercept_:', model.intercept_)
print('coef_:', model.coef_)

# ----------
x = np.array([[1,2], [1,5],[1,8]])
y= np.array([4,1,9]).reshape(-1, 1)

print("x", x)
print("y", y)

t1 = np.dot(x.T,x)
print("t1",t1)

t1_inv = np.linalg.inv(t1)
print("t1_inv",t1_inv)

t2 = np.dot(t1_inv, x.T)
final = np.dot(t2, y)
print("final",final)

#计算得到的sita和上述模型的参数是一样的