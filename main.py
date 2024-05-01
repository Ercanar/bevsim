#!/usr/bin/env python3
from dataclasses  import dataclass, field
from enum         import Enum
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

plt.ion()
plt.subplot(1, 2, 1)

# stuff for direct pyplot GUI updating, because plt.pause is actually cringe
_manager = plt.get_current_fig_manager()
assert _manager != None
_canvas = _manager.canvas

def update_gui():
    _canvas.flush_events()

@dataclass
class Species:
    age_death         : int              # life expectancy
    age_mature        : int              # age of seggsual maturity
    fertile_seasons   : [[int]]          # seasons (list of months) of segs (only one birth per season), cannot overlap
    food_consumption  : float            # food consumption per capita per day in kg
    food_penalty      : int              # penalty in reproduction bc of low food
    food_sources      : set(BiomassType) # food sources; can be other species or plants, specified to layer
    infant_mortality  : float            # infants that die in first year; 0 for None, 1 for all
    manual_dist       : dict[int, int]   # age -> group size mapping for initial population distribution
    # mass_food         : float            # mass of specimen in kg for carnivore food calculation
    n_birth           : int              # expected number of offspring per birth
    segs_probability  : float            # probability of segs in season
    starvation_time   : float            # average days to starve
    verdursten_time   : float            # average days to verdurst
    water_consumption : float            # liters of water per capita per day in l
    water_sources     : WaterSource      # see WaterSource for explanations

class Biotope(Enum):  #     boden, büsche, treetops
    Mischwald = Biomass(  500_000, 250_000, 10_000_000)
    Nadelwald = Biomass(  500_000, 250_000, 20_000_000)
    Laubwald  = Biomass(  500_000, 250_000, 15_000_000)
    Wiese     = Biomass(2_500_000,       0,          0)

@dataclass
class Environment:
    biotope_type_dist    : dict[Biotope, float] # distribution of kinds of vegetation biotopes; must add up to 1
    environment_deaths   : float                # deaths per month; 0 for None, 1 for extinktion
    fertility            : float                # fertility after environment penalty is applied
    minimum_food         : int                  # minimum amount of food during winter
    plant_growth_months  : [int]                # months where plant biomass is replenished
    simulation_area      : float                # area of simulation in km²
    simulation_time      : int                  # simulation time in years
    water_storage        : float                # reservoir of water in biotop in l (filled up by rain or river)

class Utils:
    MAX_AGE_FACTOR = 5/4 # factor for maximum age, calculated by life expectancy

    MONTHS_IN_YEAR = 12
    DAYS_IN_MONTH = 30

    @staticmethod
    def getd(xs, i, x):
        if i < len(xs):
            return xs[i]
        return x

    @staticmethod
    @cache
    def convert_probs(p_y):
        if p_y == 0:
            return 0

        if p_y <= 0.75:
            return 250 * p_y / (90000 - 49887 * p_y)
        elif p_y >= 0.998:
            return 0.05
        else:
            return np.exp( 20 / 721 * (500 * p_y - 663)) + 181 / 50000

    @staticmethod
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

    @staticmethod
    def do_death(population, age_death, max_age):
        population = population[: max_age + 2]
        splits = Utils.logistic_splits(age_death / 2 + 1)
        for i in range(len(splits)):
            population[int(i - 2/3*age_death)] *= 1 - Utils.convert_probs(splits[i])
        return population # this would be unneccessary if Python wasn't stupid and a dogshit excuse of a language

@dataclass
class Simulation:
    class AllDead(Exception):
        pass

    environment: Environment
    species:     Species

    max_age:     int = None
    max_sim_age: int = 0

    population:      [int]   = None
    population_hist: [int]   = None
    food_layers:     Biomass = None
    total_food_init: int     = None
    total_food_curr: int     = None

    verdursten_time_curr: int = None
    starvation_time_curr: int = None

    do_render: bool = True

    def __post_init__(self):
        self.max_age = int(np.ceil(Utils.MAX_AGE_FACTOR * self.species.age_death))

        self.population = np.zeros(self.max_age + 1)
        np.put(
            self.population,
            list(self.species.manual_dist.keys()),
            list(self.species.manual_dist.values()))

        self.food_layers = Biomass(*map(sum, [[
                biotope.value.__dict__[field] * factor * self.environment.simulation_area
                for (biotope, factor) in self.environment.biotope_type_dist.items()
            ] for field in Biomass.__dataclass_fields__]))

        self.total_food_init = sum([
            self.food_layers.__dict__[source.value]
            for source in self.species.food_sources])

        self.reset_food_curr()
        self.reset_verdursten()
        self.reset_starvation()

    def reset_food_curr(self):
        self.total_food_curr = self.total_food_init

    def reset_verdursten(self):
        self.verdursten_time_curr = self.species.verdursten_time

    def reset_starvation(self):
        self.starvation_time_curr = self.species.starvation_time

    # TODO cache this but NOT @cache
    # TODO enlargen brain to understand previous comment
    def psum(self):
        return np.sum(self.population)

    def run(self):
        start = time()

        try:
            for year_0 in range(self.environment.simulation_time):
                self.step_year(year_0)
        except Simulation.AllDead:
            print("oopsie woopsie all entities are ded :3")

        stop = time()
        print(f"Took {stop - start}s")

    def render(self, year_0, month):
        plt.clf()

        plt.subplot(2, 1, 1)
        plt.title(f"cute bunnygirls - Y {year_0} M {month}", fontsize = "xx-large")
        self.max_sim_age = max(max(self.population), self.max_sim_age)
        plt.xlim(0, self.max_sim_age + 50)
        plt.ylim(-1, self.max_age + 2)
        plt.xlabel("population", fontsize = "xx-large")
        plt.ylabel("age", fontsize = "xx-large")
        plt.barh(range(len(self.population)), self.population)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.population_hist)), self.population_hist)

        update_gui()

    def step_year(self, year_0):
        for month_0 in range(Utils.MONTHS_IN_YEAR):
            self.step_month(year_0, month_0)

        self.population = np.insert(self.population, 0, 0)

    def step_month(self, year_0, month_0):
        month = month_0 + 1
        if month in self.environment.plant_growth_months:
            self.reset_food_curr()

        current_fertile_season = next(filter(lambda s: month in s, self.species.fertile_seasons), None)

        for _ in range(Utils.DAYS_IN_MONTH):
            self.step_day(month, current_fertile_season)

        if self.do_render:
            self.render(year_0, month_0)

    def step_day(self, month, current_fertile_season):
        psum = self.psum()

        self.population[0] *= 1 - Utils.convert_probs(self.species.infant_mortality)

        if current_fertile_season != None and \
           self.total_food_curr - psum * self.species.food_consumption > 0:
            self.do_segs(
                Utils.gaussian_splits(len(current_fertile_season)) \
                [month - current_fertile_season[month - current_fertile_season[0]]])
            assert psum >= 0

        # andrew
        if self.environment.water_storage < self.species.water_consumption * psum:
            self.verdursten_time_curr -= 1
        else:
            self.reset_verdursten()

        # ashley
        if self.total_food_curr < self.species.food_consumption * psum:
            self.starvation_time_curr -= 1
        else:
            self.reset_starvation()

        for f in \
        [ self.do_starve
        , self.do_thirst
        , self.do_widespread_industrial_sabotage_uwu ]:
            old_psum = self.psum()
            f()
            psum = self.psum()
            assert psum >= 0

        self.population_hist = np.append(self.population_hist, psum)
        self.population = Utils.do_death(self.population, self.species.age_death, self.max_age)
        self.population = np.floor(self.population)
        if self.psum() == 0:
            raise Simulation.AllDead()

        consumption = self.species.food_consumption * psum
        self.total_food_curr = max(
                self.environment.minimum_food,
                self.total_food_curr - consumption)

    def do_segs(self, gauss_factor):
        fertile = np.sum(self.population[self.species.age_mature:]) / 2
        births = np.floor(
            self.environment.fertility
            * self.species.segs_probability
            * self.species.n_birth
            * fertile
            * gauss_factor
            * ((1 - self.psum() * self.species.food_consumption / self.total_food_curr)) ** self.species.food_penalty)
        self.population[0] += births / 30

    def do_starve(self):
        if self.starvation_time_curr >= 0:
            return

        psum = self.psum()
        starving_population = self.total_food_curr / self.species.food_consumption - psum
        self.population = np.vectorize(lambda g:
            g + starving_population * g / psum * Utils.getd(
                Utils.logistic_splits(self.species.starvation_time),
                -self.starvation_time_curr - 1, 1))(self.population)

    def do_thirst(self):
        if self.species.water_sources == WaterSource.Implicit \
           or self.verdursten_time_curr >= 0:
               return

        psum = self.psum()
        split = Utils.logistic_splits(self.species.water_consumption)
        tmp = (psum * self.species.water_consumption - self.environment.water_storage) / self.species.water_consumption
        deaths = (split[-self.verdursten_time_curr - 1] \
                  if len(split) >= -self.verdursten_time_curr
                  else 1) * tmp

        self.population -= deaths // len(self.population) + 1

    def do_widespread_industrial_sabotage_uwu(self):
        self.population *= 1 - self.environment.environment_deaths / Utils.DAYS_IN_MONTH

################################################################################

environment = Environment(
    biotope_type_dist    = {Biotope.Mischwald: 0.65, Biotope.Wiese: 0.35},
    environment_deaths   = 0.005,
    fertility            = 0.999,
    minimum_food         = 1000,
    plant_growth_months  = [3, 4, 5, 6, 7, 8, 9, 10],
    simulation_area      = 1,
    simulation_time      = 2,
    water_storage        = 1_000_000,
)

bunnies = Species(
    age_death         = 12,
    age_mature        = 1,
    fertile_seasons   = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    food_consumption  = 1.35,
    food_penalty      = 500,
    food_sources      = set([BiomassType.Ground, BiomassType.Bushes]),
    infant_mortality  = 0.3,
    manual_dist       = {4: 300, 1: 20},
    # mass_food         = 4.5,
    n_birth           = 3,
    segs_probability  = 0.999,
    starvation_time   = 14,
    verdursten_time   = 14,
    water_consumption = 0,
    water_sources     = WaterSource.Implicit,
)

sim = Simulation(environment, bunnies)
# sim.do_render = False
print("HERE GO HERE PLEASE =============================")
sim.run()
plt.ioff()
plt.show()

# we stay silly :3
