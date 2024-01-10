#!/usr/bin/env python3
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from preset1 import *
from lookup import *
from environment1 import *
from time import time
from myTypes import *

time_steps = 64 # years of simulation

def herbivore_food(get):
    return sum([
        np.array(get(plant_biomass_calc[name]) * factor * biotop_area)
        for (name, factor) in biotop_type_dist.items()])

plant_masses = Biomass(*[
    herbivore_food(lambda b: b.__dict__[field])
    for field in Biomass.__dataclass_fields__.keys()])

def gaussian_func(x):
        return (np.sqrt(2 * np.pi)) ** -1 * np.exp(-(x**2) / 2)

def logistic_splits(n):
    splits = np.array([(i + 1) * 8 / int(n) - 4 for i in range(int(n) - 1)])
    splits = np.append(splits, sc.inf)
    probs  = [
        sc.integrate.quad(gaussian_func, -sc.inf, splits[i])[0]
        for i in range(len(splits))
    ]
    probs.insert(0, 0)
    probs[-1] = 1
    return [(probs[i + 1] + probs[i]) / 2 for i in range(len(probs) - 1)]

LogisticSplits = Memo(logistic_splits)

def gaussian_splits(n):
    if n == 0:
        return np.array([0])
    splits = np.array([(i + 1) * 8 / int(n) - 4 for i in range(int(n) - 1)])
    splits = np.append(splits, sc.inf)
    splits = np.insert(splits, 0, -sc.inf)
    probs  = [
            sc.integrate.quad(gaussian_func, splits[i], splits[i+1])[0]
            for i in range(len(splits) - 1)]
    return probs

GaussianSplits = Memo(gaussian_splits)

def max_age():
    return int(np.ceil(5 / 4 * age_death))


def segs(x, gauss_factor, food):
    fertile = sum(x[4:])
    births = np.floor(
        fertility
        * species_factor
        * n_birth
        * fertile
        / 2
        * gauss_factor
        * np.sqrt((1 - sum(x) * food_consumption / food))
    )
    x[0] = x[0] + births / 30
    return x

def thirsty(x):
    return water_storage < water_consumption * sum(x)

def starvy(x, food):
    return food < food_consumption * sum(x)

class Death:
    def food(x, food, timer, t2):
        if t2 < 0:
            if len(LogisticSplits(timer)) >= -t2:
                deaths =  LogisticSplits(timer)[-t2-1] * (sum(x) * water_consumption - water_storage) / (water_consumption)
            else:
                deaths = (sum(x) * water_consumption - water_storage) / (water_consumption)
            return x - deaths // len(x) - 1
        else:
            return x

    def accidents(x):
        return (1 - environment_deaths / 30000 - 1/30000) * np.array(x)

    def age(x):
        x = x[: max_age()]
        for i in range(len(LogisticSplits(age_death / 2))):
            x[-i] = x[-i] * LogisticSplits(age_death / 2)[i]
        return x

    def thirst(x, source, timer, t1):
        if source == WaterSource.Implicit:
            return x
        elif t1 < 0:
            if len(LogisticSplits(timer)) >= -t1:
                deaths =  LogisticSplits(timer)[-t1-1] * (sum(x) * water_consumption - water_storage) / (water_consumption)
            else:
                deaths = (sum(x) * water_consumption - water_storage) / (water_consumption)
            return x - deaths // len(x) - 1
        else:
            return x


# TODO funktion fuer simulation, inputs: array von liste an presets, available food, etc

bev = sum(initial_distribution)
bev_hist = np.array([bev])
bev_distribution = initial_distribution
t1 = verdursten_time
t2 = starvation_time
plant_mass_sum = sum([plant_masses(s) for s in food_sources])

start = time()
for _ in range(time_steps): # yearly simulation
    bev_distribution = Death.age(bev_distribution)
    for i in range(1, 13): # monthly simulation
        if i in plant_growth_months:
            plant_mass_sum = sum([plant_masses(s) for s in food_sources])
        foo = next(
            filter(
                lambda a: i in a[0], zip(fertile_months, range(len(fertile_months)))
            ),
            (0, 0),
        )
        for _ in range(1, 31): # daily simulation
            if foo[0] == 0:
                bev_distribution = bev_distribution
            else:
                if plant_mass_sum - sum(bev_distribution) * food_consumption > 0:
                    bev_distribution = segs(
                        bev_distribution,
                        GaussianSplits(len(foo[0]))[i - foo[0][i - foo[0][0]]],
                        # gaussian_splits(len(foo[0]))[i - foo[0][0]], # TODO replace gaussian lookup with actual maths
                        plant_mass_sum,
                    )
            if thirsty(bev_distribution) == True:
                t1 -= 1
            else:
                t1 = verdursten_time
            if starvy(bev_distribution, plant_mass_sum) == True:
                t2 -= 1
            else:
                t2 = starvation_time
            bev_distribution = Death.food(bev_distribution, plant_mass_sum, starvation_time, t2)
            if plant_mass_sum - food_consumption * sum(bev_distribution) < 0:
                plant_mass_sum = 0
            else :
                plant_mass_sum -= food_consumption * sum(bev_distribution)
            bev_distribution = Death.thirst(bev_distribution, water_sources, verdursten_time, t1)
            bev_distribution = Death.accidents(bev_distribution)
            bev_hist = np.append(bev_hist, sum(bev_distribution))
    bev_distribution[0] = bev_distribution[0] * (1 - infant_mortality / 1000)
    bev_distribution = np.insert(bev_distribution, 0, 0)

print(f"Took {time() - start}s")

plt.plot(range(len(bev_hist)), bev_hist)
plt.xlabel("time in days", fontsize="xx-large")
plt.ylabel("bev in #", fontsize="xx-large")
plt.grid()
plt.show()
