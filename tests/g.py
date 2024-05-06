#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import lmaofit

ys = 1 - np.array([100, 97.2, 97.2, 94.4, 93.1, 91.7, 89.8, 88.5, 86.8, 85.5, 85.3, 82.6, 81.3, 80, 78.6, 77.4, 76, 74.8, 73.6, 72.5, 71.3])/100
xs = np.array([0, 3.5, 7, 10.2, 13.4, 16.5, 19.4, 22.3, 25, 27.7, 30.7, 32.7, 35.1, 37.4, 39.6, 41.8, 43.8, 45.8, 47.7, 49.6, 51.3]) /100

@lmaofit.wrap
def ff(x, y, m, n):
    return x * m + n - y

res = lmaofit.fit(ff, xs, ys)
lmaofit.plot(ff, res, np.concatenate([[0], xs]), color = "black", label = "Fitgerade")

plt.scatter(xs, ys)
plt.xlabel("wahrscheinlichkeit nach 360 tagen", fontsize="xx-large")
plt.ylabel("genauigkeit des linearen modells", fontsize="xx-large")
plt.legend()
plt.show()
