import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0, 2, 2, 2, 4]).reshape((-1, 1))
y = np.array([1, 1, 2, 3, 3])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)

print('intercept_:', model.intercept_)
print('coef_:', model.coef_)


y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

import matplotlib.pyplot as plt
import numpy as np

plt.figure(num = 1, figsize=(8, 5))
plt.plot(x, y)
plt.plot(x, y_pred, color='red', linewidth=1.0, linestyle='--' )
plt.show()

# ------------------------------
x = np.array([2,5,8]).reshape((-1, 1))
y = np.array([4,1,9])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)

print('intercept_:', model.intercept_)
print('coef_:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

plt.figure(num = 2, figsize=(8, 5))
plt.plot(x, y)
plt.plot(x, y_pred, color='red', linewidth=1.0, linestyle='--' )
plt.show()
