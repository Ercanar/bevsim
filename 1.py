#!/usr/bin/env python3
import numpy             as np
import scipy             as sc
import matplotlib.pyplot as plt
from preset1      import *
from lookup       import *
from environment1 import *

time_steps     = 83 # years of simulation

def herbivore_food(x):
    return x #TODO: return touples of eg [("ground", 100), ("bushes", 50), ("treetops", 200)]

plant_mass = herbivore_food(biotop_type_dist)

def logistic_splits(n):
    def gaussian_func(x):
        return (np.sqrt(2*np.pi)) ** -1 * np.exp(-(x**2) / 2)
    splits = np.array([(i+1)*8/int(n) - 4 for i in range(int(n)-1)])
    splits = np.append(splits, sc.inf)
    probs  = [sc.integrate.quad(gaussian_func, -sc.inf, splits[i])[0] for i in range(len(splits))]
    probs.insert(0, 0)
    probs[-1] = 1
    return [(probs[i+1] + probs[i])/2 for i in range(len(probs) - 1)]

def max_age():
    return int(np.ceil(5/4 * age_death))

def segs(x, gauss_factor, food):
    fertile = sum(x[4:])
    births = np.floor(fertility * species_factor * n_birth * fertile / 2 * gauss_factor * np.sqrt((1 - sum(x) * food_consumption / food)))
    x[0] = x[0] + births
    return x

class Death:
    def food(x, food):
        if food - sum(x) * food_consumption < 0:
            counter_after_deaths = sum(x) - (sum(x) * food_consumption - food) / (2 * food_consumption)
            return x - (sum(x) - counter_after_deaths) // len(x)

        else:
            return x

    def accidents_month(x):
        return (999 - environment_deaths)/1000 * np.array(x)

    def age(x):
        x = x[:max_age()]
        for i in range(len(logistic_splits(age_death/2))):
            x[-i] = x[-i] * logistic_splits(age_death/2)[i]
        return x

#TODO funktion fuer simulation, inputs: array von liste an presets, available food, 

bev = sum(initial_distribution)
bev_hist = np.array([bev])
bev_distribution = initial_distribution

for _ in range(time_steps):
    bev_distribution = Death.age(bev_distribution)
    for i in range(1, 13):
        bev_distribution = Death.food(bev_distribution, 100000)
        bev_distribution = Death.accidents_month(bev_distribution)
        foo = next(filter(
            lambda a: i in a[0],
            zip(fertile_months, range(len(fertile_months)))
        ), (0, 0))
        if foo[0] == 0:
            bev_distribution = bev_distribution
        else:
            if 100000 - sum(bev_distribution) * food_consumption > 0:
                bev = segs(bev_distribution, gaussian_splits[len(foo[0])][i - foo[0][i - foo[0][0]]], 100000) #TODO replace gaussian splits in lookup with actual maths...
        bev_distribution[0] = bev_distribution[0] * (1 - infant_mortality/1000)
        bev_hist = np.append(bev_hist, sum(bev_distribution))
    bev_distribution = np.insert(bev_distribution, 0, 0)


plt.plot(range(len(bev_hist)), bev_hist)
plt.xlabel("time in months", fontsize = "xx-large")
plt.ylabel("bev in #"      , fontsize = "xx-large")
plt.grid()
plt.show()
