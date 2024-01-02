#!/usr/bin/env python3
import numpy             as np
import matplotlib.pyplot as plt
from preset1      import *
from lookup       import *
from environment1 import *

time_steps = 25 # years of simulation

def max_age():
    return 5/4 * age_death

def segs(x, gauss_factor):
    return (x + fertility * species_factor * n_birth * x / 2 * gauss_factor * (1 - x / food))

class Death:
    def food(x):
        if food - x < 0:
            return x - 0.5 * abs(food - x)
        else:
            return x

    def desease_month(x):
        return 999/1000 * x

    def age(x):
        return x

bev = initial
bev_hist = np.array([bev])
bev_distribution = initial_distribution

for _ in range(time_steps):
    for i in range(1, 13):
        bev = Death.food(bev)
        bev = Death.desease_month(bev)
        foo = next(filter(
            lambda a: i in a[0],
            zip(fertile_months, range(len(fertile_months)))
        ), (0, 0))
        if foo[0] == 0:
            bev = bev
        else:
            if food - bev > 0:
                bev = segs(bev, gaussian_splits[len(foo[0])][i - foo[0][i - foo[0][0]]])
        bev_hist = np.append(bev_hist, bev)

print(len(initial_distribution))
print(max_age())
exit(42)

plt.plot(range(len(bev_hist)), bev_hist)
plt.xlabel("time in months", fontsize = "xx-large")
plt.ylabel("bev in #"      , fontsize = "xx-large")
plt.grid()
plt.show()
