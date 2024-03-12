#!/usr/bin/env python3
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from preset1 import *
from lookup import *
from environment1 import *
from time import time
from myTypes import *

def thirsty(x): # andrew
    return water_storage < water_consumption * sum(x)

def starvy(x, food): # ashley
    return food < food_consumption * sum(x)

def getd(xs, i, x):
    if i < len(xs):
        return xs[i]
    return x

cutoff = 0

def debug(lbl, x):
    print(lbl, x)
    return x

class Death:
    def food(bev_distribution, food, timer, t2):
        if t2 >= 0:
            return bev_distribution

        total = sum(bev_distribution)
        print("total:", total)
        starving_people = food / food_consumption - total
        res = list(map(
            lambda g: g + debug("starve:", starving_people * g / total * \
                          getd(LogisticSplits(timer), -t2-1, 1)),
            bev_distribution))

        global cutoff
        cutoff += 1
        if cutoff >= 20:
            exit()
        return res

    def accidents(x):
        return (1 - environment_deaths / 30000 - 1/30000) * np.array(x)

    def thirst(x, source, timer, t1):
        if source == WaterSource.Implicit or t1 >= 0:
            return x

        if len(LogisticSplits(timer)) >= -t1:
            deaths =  LogisticSplits(timer)[-t1-1] * (sum(x) * water_consumption - water_storage) / water_consumption
        else:
            deaths = (sum(x) * water_consumption - water_storage) / water_consumption

        return x - deaths // len(x) - 1


# TODO funktion fuer simulation, inputs: array von liste an presets, available food, etc

bev = sum(initial_distribution)
bev_hist = np.array([bev])
bev_distribution = initial_distribution
t1 = verdursten_time
t2 = starvation_time
start = time()
for _ in range(time_steps): # yearly simulation
    for i in range(1, 13): # monthly simulation
        for _ in range(30): # daily simulation
            if thirsty(bev_distribution):
                t1 -= 1
            else:
                t1 = verdursten_time

            if starvy(bev_distribution, plant_mass_sum):
                t2 -= 1
            else:
                t2 = starvation_time

            print(plant_mass_sum)
            bev_distribution = Death.food(bev_distribution, plant_mass_sum, starvation_time, t2)
            assert sum(bev_distribution) > 0

            if plant_mass_sum - food_consumption * sum(bev_distribution) < 0:
                plant_mass_sum = 0
            else:
                plant_mass_sum -= food_consumption * sum(bev_distribution)

            bev_distribution = Death.thirst(bev_distribution, water_sources, verdursten_time, t1)
            assert sum(bev_distribution) > 0

            bev_distribution = Death.accidents(bev_distribution)
            assert sum(bev_distribution) > 0

            bev_hist = np.append(bev_hist, sum(bev_distribution))

print(f"Took {time() - start}s")

plt.plot(range(len(bev_hist)), bev_hist)
plt.xlabel("time in days", fontsize="xx-large")
plt.ylabel("bev in #", fontsize="xx-large")
plt.grid()
plt.show()
