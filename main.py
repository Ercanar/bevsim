#!/usr/bin/env python3
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from lookup import *
from environment1 import *
from time import time
from myTypes import *
from dataclasses import dataclass
from functools import cache

WaterSource = Enum("WaterSource", [
    "Implicit", # water is contained in food
    "Explicit", # water needs to be sourced from environment
])

@dataclass
class Species:
    age_death         : int              # life expectancy
    age_mature        : int              # age of seggsual maturity
    fertile_seasons   : [[int]]          # seasons (list of months) of segs (only one birth per season), cannot overlap
    food_consumption  : float            # food consumption per capita per day in kg
    food_sources      : set(BiomassType) # TODO implement carnivores/omnivores
    infant_mortality  : float            # promille of infants that die in first year
    manual_dist       : dict[int, int]   # age -> group size mapping for initial population distribution
    # mass_food         : float            # mass of specimen in kg for carnivore food calculation
    n_birth           : int              # expected number of offspring per birth
    segs_probability  : float            # probability of segs in season
    starvation_time   : float            # average days to starve
    verdursten_time   : float            # average days to verdurst
    water_consumption : float            # liters of water per capita per day in l
    water_sources     : WaterSource      # see WaterSource for explanations

bunnies = Species(
    age_death         = 12,
    age_mature        = 1,
    fertile_seasons   = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    food_consumption  = 1.35,
    food_sources      = set([BiomassType.Ground, BiomassType.Bushes]),
    infant_mortality  = 300,
    manual_dist       = {4: 300},
    # mass_food         = 4.5,
    n_birth           = 3,
    species_factor    = 0.999,
    starvation_time   = 14,
    verdursten_time   = starvation_time,
    water_consumption = 0,
    water_sources     = WaterSource.Implicit,
)

class Utils:
    MAX_AGE_FACTOR = 5/4

    # initial age group sizes pre-simulation
    initial_distribution: [float] # numpy

    def __init__(self, species: Species):
        self.initial_distribution = np.zeros(int(np.ceil(self.MAX_AGE_FACTOR * species.age_death)) + 1)
        np.put(self.initial_distribution, species.manual_dist[0], species.manual_dist[1])

utils = Utils(bunnies)

