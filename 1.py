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

    def age_and_desease_year(x):
        return 0.98 * x

    def age_and_desease_month(x):
        return 599/600 * x

bev = initial
bev_hist = np.array([bev])

for _ in range(time_steps):
    i = 1
    for i < 13:
        bev = Death.food(bev)
        bev = Death.age_and_desease_month(bev)
        if i in fertile_months:
            pass
        #gauss ueber die fertile months segs(bev) if food - bev > 0
        else:
            pass
        i += 1
        bev_hist = np.append(bev_hist, bev)

plt.plot(range(time_steps * 12 + 1), bev_hist)
plt.show()
