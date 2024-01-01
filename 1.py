#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from preset1 import *

time_steps = 25

def segs(x):
    return (x + n_birth * x / 2) * n_year

class Death:
    def food(x):
        if food - x < 0:
            return x - abs(food - x)
        else:
            return x

    def age_and_desease(x):
        return 0.98 * x

bev = initial
bev_hist = np.array([bev])

for _ in range(time_steps):
    bev = Death.food(bev)
    if food - bev > 0:
        bev = segs(bev)
    bev = Death.age_and_desease(bev)
    bev_hist = np.append(bev_hist, bev)

plt.plot(range(time_steps + 1), bev_hist)
plt.show()
