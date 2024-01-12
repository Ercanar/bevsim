#!/usr/bin/env python3
from dataclasses  import dataclass, field
from enum         import Enum
from environment1 import *
from functools    import cache
from time         import time
import matplotlib.pyplot as plt
import numpy             as np
import scipy             as sc

fields = ["Ground", "Bushes", "Treetop"]
BiomassType = Enum("BiomassType", [[f, f] for f in fields])
Biomass = dataclass(type("Biomass", (object,), {
    "__call__": (lambda self, val: self.__dict__[val.value]),
    "__annotations__": {field: int for field in fields}}))
del fields

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
    food_sources      : set(BiomassType) # food sources; can be other species or plants, specified to layer
    infant_mortality  : float            # promille of infants that die in first year
    manual_dist       : dict[int, int]   # age -> group size mapping for initial population distribution
    # mass_food         : float            # mass of specimen in kg for carnivore food calculation
    n_birth           : int              # expected number of offspring per birth
    segs_probability  : float            # probability of segs in season
    starvation_time   : float            # average days to starve
    verdursten_time   : float            # average days to verdurst
    water_consumption : float            # liters of water per capita per day in l
    water_sources     : WaterSource      # see WaterSource for explanations

class Biotope(Enum):
    Mischwald = Biomass(  500_000, 250_000, 10_000_000)
    Nadelwald = Biomass(  500_000, 250_000, 20_000_000)
    Laubwald  = Biomass(  500_000, 250_000, 15_000_000)
    Wiese     = Biomass(1_000_000,       0,          0) # wert ist für trockenes weed

@dataclass
class Environment:
    simulation_area      : float                # area of simulation in km²
    biotope_type_dist    : dict[Biotope, float] # distribution of kinds of vegetation biotopes; must add up to 1
    environment_deaths   : float                # promille deaths per month; 999 for extinktion
    fertility            : float                # fertility after environment penalty is applied
    plant_growth_months  : [int]                # months where plant biomass is replenished
    simulation_time      : int                  # simulation time in years
    water_storage        : float                # reservoir of water in biotop in l (filled up by rain or river)

environment = Environment(
    simulation_area      = 1,
    biotope_type_dist    = {Biotope.Mischwald: 0.65, Biotope.Wiese: 0.35},
    environment_deaths   = 1,
    fertility            = 0.999,
    plant_growth_months  = [3, 4, 5, 6, 7, 8, 9, 10],
    simulation_time      = 64,
    water_storage        = 1_000_000,
)

bunnies = Species(
    age_death         = 12,
    age_mature        = 1,
    fertile_seasons   = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    food_consumption  = 1.35,
    food_sources      = set([BiomassType.Ground, BiomassType.Bushes]),
    infant_mortality  = 300,
    manual_dist       = {4: 300, 1: 20},
    # mass_food         = 4.5,
    n_birth           = 3,
    segs_probability  = 0.999,
    starvation_time   = 14,
    verdursten_time   = 14, # TODO this is sus; requesting cleanup
    water_consumption = 0,
    water_sources     = WaterSource.Implicit,
)

@dataclass()
class Utils:
    MAX_AGE_FACTOR = 5/4 # factor for maximum age, calculated by life expectancy

    species: Species
    environment: Environment

    # initial age group sizes pre-simulation
    initial_distribution: [float] = None # numpy

    herbivore_food: Biomass = None # total available plant mass, divided in layers
    species_food: int = None # total available food for specific species

    def __post_init__(self):
        self.initial_distribution = np.zeros(int(np.ceil(self.MAX_AGE_FACTOR * self.species.age_death)) + 1)
        np.put(
            self.initial_distribution,
            list(self.species.manual_dist.keys()),
            list(self.species.manual_dist.values()))

        self.herbivore_food = Biomass(*map(sum, [[
                biotope.value.__dict__[field] * factor * self.environment.simulation_area
                for (biotope, factor) in self.environment.biotope_type_dist.items()
            ] for field in Biomass.__dataclass_fields__]))

        self.species_food = sum([
            self.herbivore_food.__dict__[source.value]
            for source in self.species.food_sources])

    def gaussian_func(x):
        return (np.sqrt(2 * np.pi)) ** -1 * np.exp(-(x**2) / 2)

    @staticmethod
    @cache
    def logistic_splits(n):
        splits = np.array([(i + 1) * 8 / int(n) - 4 for i in range(int(n) - 1)])
        splits = np.append(splits, sc.inf)

        probs = [
            sc.integrate.quad(Utils.gaussian_func, -sc.inf, splits[i])[0]
            for i in range(len(splits))
        ]

        probs.insert(0, 0)
        probs = [(probs[i + 1] + probs[i]) / 2 for i in range(len(probs) - 1)]
        probs[-1] = 1
        return probs

    @staticmethod
    @cache
    def gaussian_splits(n):
        if n == 0:
            return np.array([0])

        splits = np.array([(i + 1) * 8 / int(n) - 4 for i in range(int(n) - 1)])
        splits = np.append(splits, sc.inf)
        splits = np.insert(splits, 0, -sc.inf)

        probs = [
            sc.integrate.quad(Utils.gaussian_func, splits[i], splits[i+1])[0]
            for i in range(len(splits) - 1)]

        return probs

utils = Utils(bunnies, environment)
