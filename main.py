#!/usr/bin/env python3
from copy         import deepcopy
from dataclasses  import dataclass, field
from enum         import Enum
from functools    import cache
from random       import random
from time         import time
from typing       import cast, Any, Callable, Optional
import math
import matplotlib.pyplot as plt
import numpy             as np
import numpy.typing      as npt
import scipy             as sc # type: ignore

F = Callable[..., Any]
Floats = npt.NDArray[np.float64]

# BiomassType = Enum("BiomassType", ["Ground", "Bushes", "Treetop"])
class BiomassType(Enum):
    Ground = "Ground"
    Bushes = "Bushes"
    Treetop = "Treetop"

@dataclass
class Biomass():
    Ground  : int
    Bushes  : int
    Treetop : int

WaterSource = Enum("WaterSource", [
    "Implicit", # water is contained in food
    "Explicit", # water needs to be sourced from environment
])

plt.ion()

def lazy(f: F) -> F:
    return lambda *args, **kwargs: lambda: f(*args, **kwargs)

@dataclass
class Species:
    name: str

    age_death         : int              # life expectancy
    age_mature        : int              # age of seggsual maturity
    fertile_seasons   : list[list[int]]  # seasons (list of months) of segs (only one birth per season), cannot overlap
    food_consumption  : float            # food consumption per capita per day in kg
    food_sources      : set[BiomassType] # food sources; can be other species or plants, specified to layer
    infant_mortality  : float            # infants that die in first year; 0 for None, 1 for all
    manual_dist       : dict[int, int]   # age -> group size mapping for initial population distribution
    # mass_food         : float            # mass of specimen in kg for carnivore food calculation
    n_birth           : int              # expected number of offspring per birth
    segs_probability  : float            # probability of segs in season
    starvation_time   : float            # average days to starve
    verdursten_time   : float            # average days to verdurst
    water_consumption : float            # liters of water per capita per day in l
    water_sources     : WaterSource      # see WaterSource for explanations

    # late constants
    max_age      : int = 0 # life expectancy adjusted with MAX_AGE_FACTOR
    max_age_curr : int = 0

    # simulation state variables

    population      : Floats = field(default_factory = lambda: np.zeros(0))
    population_hist : Floats = field(default_factory = lambda: np.zeros(0))

    total_food_init      : float = 0
    verdursten_time_curr : float = 0
    starvation_time_curr : float = 0

    current_fertile_season : Optional[list[int]] = None

    def __post_init__(self) -> None:
        self.max_age = int(np.ceil(Utils.MAX_AGE_FACTOR * self.age_death))

        self.population = np.zeros(self.max_age + 1)
        np.put(
            self.population,
            list(self.manual_dist.keys()),
            list(self.manual_dist.values()))

        self.population_hist = np.array([])

    def reset_verdursten(self) -> None:
        self.verdursten_time_curr = self.verdursten_time

    def reset_starvation(self) -> None:
        self.starvation_time_curr = self.starvation_time

    def update_current_fertile_season(self, month: int) -> None:
        self.current_fertile_season = next(filter(
            lambda s: month in s, # type: ignore
            self.fertile_seasons), None)

    def psum(self) -> float:
        return float(np.sum(self.population))

    @lazy
    def do_starve(self, available_food: int) -> None:
        if self.starvation_time_curr >= 0:
            return

        psum = self.psum()
        starving_population = (available_food / self.food_consumption - psum) / len(self.population)
        self.population = np.vectorize(lambda g:
            g + starving_population * Utils.getd(
                0.5 * Utils.logistic_splits(self.starvation_time),
                int(-self.starvation_time_curr), 0.5))(self.population)

    @lazy
    def do_thirst(self, water_storage: float) -> None:
        if self.water_sources == WaterSource.Implicit \
           or self.verdursten_time_curr >= 0:
               return

        psum = self.psum()
        split = Utils.logistic_splits(int(self.water_consumption))
        tmp = (psum * self.water_consumption - water_storage) / self.water_consumption
        deaths = (split[int(-self.verdursten_time_curr - 1)] \
                  if len(split) >= -self.verdursten_time_curr
                  else 1) * tmp

        self.population -= deaths // len(self.population) + 1

    @lazy
    def do_widespread_industrial_sabotage_uwu(self, environment_deaths: float) -> None:
        self.population = np.array(self.population) * (1 - Utils.convert_probs(environment_deaths))

    def do_segs(
        self,
        fertility           : float,
        gauss_factor        : float,
        available_food      : float,
        plant_growth_months : list[int],
        month               : int
    ) -> None:
        psum = self.psum()
        fertile = np.sum(self.population[self.age_mature:]) / 2
        births = np.floor(
            fertility
            * self.segs_probability
            * self.n_birth
            * fertile
            * gauss_factor
            * (((1 - psum * self.food_consumption / available_food))
                ** Utils.food_penalty(
                    psum,
                    self.food_consumption,
                    available_food,
                    self.total_food_init,
                    plant_growth_months,
                    month)))
        if births < 0.1 * psum: # deviants
            births += random() / 75 * fertile
        self.population[0] += births / 30


class Biotope(Enum):  #     boden, büsche, treetops
    Mischwald = Biomass(  500_000, 250_000, 10_000_000)
    Nadelwald = Biomass(  500_000, 250_000, 20_000_000)
    Laubwald  = Biomass(  500_000, 250_000, 15_000_000)
    Wiese     = Biomass(2_500_000,       0,          0)

@dataclass
class Environment:
    biotope_type_dist    : dict[Biotope, float] # distribution of kinds of vegetation biotopes; must add up to 1
    environment_deaths   : float                # deaths per year; 0 for None, 1 for extinktion
    fertility            : float                # fertility after environment penalty is applied
    minimum_food         : int                  # minimum amount of food during winter
    plant_growth_months  : list[int]            # months where plant biomass is replenished, must be continous
    simulation_area      : float                # area of simulation in km²
    simulation_time      : int                  # simulation time in years
    water_storage        : float                # reservoir of water in biotop in l (filled up by rain or river)
    water_replenish      : float                # monthly fillup of water reservoir

class Utils:
    MAX_AGE_FACTOR = 5/4 # factor for maximum age, calculated by life expectancy

    MONTHS_IN_YEAR : int = 12
    DAYS_IN_MONTH  : int = 30

    @staticmethod
    def getd(xs: Floats, i: int, x: float) -> float:
        if i < len(xs):
            return float(xs[i])
        return x

    @staticmethod
    @cache
    def convert_probs(p_y: float) -> float:
        return \
            0                                 if p_y == 0     else \
            250 * p_y / (90000 - 49887 * p_y) if p_y <= 0.75  else \
            0.05                              if p_y >= 0.998 else \
            math.exp(20 / 721 * (500 * p_y - 663)) + 181 / 50000

    @staticmethod
    def gaussian_func(x: float) -> float:
        return (math.sqrt(2 * np.pi)) ** -1 * math.exp(-(x**2) / 2)

    @staticmethod
    @cache
    def logistic_splits(n: int) -> Floats:
        splits = np.array([(i + 1) * 8 / n - 4 for i in range(n - 1)])
        splits = np.append(splits, sc.inf)

        probs = [
            sc.integrate.quad(Utils.gaussian_func, -sc.inf, splits[i])[0]
            for i in range(len(splits))
        ]

        probs.insert(0, 0)
        probs = [(probs[i + 1] + probs[i]) / 2 for i in range(len(probs) - 1)]
        probs[-1] = 1
        return np.array(probs)

    @staticmethod
    @cache
    def gaussian_splits(n : int) -> Floats:
        if n == 0:
            return np.array([0])

        splits = np.array([(i + 1) * 8 / n - 4 for i in range(n - 1)])
        splits = np.append(splits, sc.inf)
        splits = np.insert(splits, 0, -sc.inf)

        probs = [
            sc.integrate.quad(Utils.gaussian_func, splits[i], splits[i+1])[0]
            for i in range(len(splits) - 1)]

        return np.array(probs)

    @staticmethod
    def food_penalty(
        psum                : float,
        food_consumption    : float,
        total_food_curr     : float,
        total_food_init     : float,
        plant_growth_months : list[int],
        month               : int
    ) -> float:
        until_next_pgm = (min(plant_growth_months) - 1 - month) % Utils.MONTHS_IN_YEAR
        available_food = total_food_curr + total_food_init * until_next_pgm
        if until_next_pgm == 0:
            z = total_food_curr
        elif until_next_pgm >= 12: # prevent mass extinctin bcuz of horny winter uwu
            z = available_food
        else:
            z = available_food / until_next_pgm
        y = z - Utils.MONTHS_IN_YEAR * Utils.DAYS_IN_MONTH * food_consumption * psum
        if y >= 0:
            y = 0
        return - y 

    @staticmethod
    def do_death(population: Floats, age_death: int, max_age: int) -> Floats:
        population = population[: max_age + 2]
        splits = Utils.logistic_splits(int(age_death / 2 + 2))
        splits = 1 - (splits / len(splits)) ** 2
        population = np.array(list(map(
            lambda x, y: x * y, population,
            [1] * (len(population) - len(splits)) + list(splits))))
        return population

@dataclass
class Simulation:
    class AllDead(Exception):
        pass

    environment: Environment
    species:     list[Species]

    gui_top : list[Any] = field(default_factory = lambda: [])
    gui_bot : Any       = None

    max_sim_age: int = 0

    food_layers_init : Biomass = field(default_factory = lambda: Biomass(0, 0, 0))
    food_layers_curr : Biomass = field(default_factory = lambda: Biomass(0, 0, 0))
    total_water_init : float   = 0
    total_water_curr : float   = 0

    do_render: bool = True

    def __post_init__(self) -> None:
        gui = plt.gcf().subfigures(2, 1)
        self.gui_top = gui[0].subplots(1, len(self.species))
        self.gui_bot = gui[1].subplots(1, 1)

        self.food_layers_init = Biomass(*map(sum, [[
                biotope.value.__dict__[field] * factor * self.environment.simulation_area
                for (biotope, factor) in self.environment.biotope_type_dist.items()
            ] for field in Biomass.__dataclass_fields__]))
        self.reset_food_layers()

        for species in self.species:
            species.total_food_init = self.food_for_species(species)
            species.reset_verdursten()
            species.reset_starvation()

        self.total_water_init = self.environment.water_storage
        self.reset_water_curr()

    def reset_food_layers(self) -> None:
        self.food_layers_curr = deepcopy(self.food_layers_init)

    def reset_water_curr(self) -> None:
        self.total_water_curr = self.total_water_init

    # TODO potential performance hit
    def food_groups_for_species(self, species: Species) -> dict[BiomassType, float]:
        return {
            source: self.food_layers_curr.__dict__[source.value]
            for source in species.food_sources}

    def food_for_species(self, species: Species) -> float:
        return sum(cast(list[float], self.food_groups_for_species(species).values()))

    def run(self) -> None:
        start = time()

        try:
            for year_0 in range(self.environment.simulation_time):
                self.step_year(year_0)
        except Simulation.AllDead:
            print("oopsie woopsie all entities are ded :3")

        stop = time()
        print(f"Took {stop - start}s")

    def render(self, year_0: int, month: int) -> None:
        self.gui_bot.clear()

        for (i, species) in enumerate(self.species):
            fig = self.gui_top[i]
            fig.clear()
            species.max_age_curr = max(max(species.population), species.max_age_curr)
            fig.set_title(f"{species.name} - Y {year_0} M {month}", fontsize = "xx-large")
            fig.set_xlim(0, species.max_age_curr + 50)
            fig.set_ylim(-1, species.max_age + 2)
            fig.set_xlabel("population", fontsize = "xx-large")
            fig.set_ylabel("age", fontsize = "xx-large")
            fig.barh(range(len(species.population)), species.population)

            fig = self.gui_bot
            fig.set_title("total population", fontsize = "xx-large")
            fig.set_xlabel("time in days", fontsize = "xx-large")
            fig.set_ylabel("bev in #", fontsize = "xx-large")
            fig.set_yscale("log")
            fig.plot(
                range(len(species.population_hist)),
                species.population_hist,
                label = species.name
            )

        plt.legend()
        plt.pause(0.0000000001)

    def step_year(self, year_0: int) -> None:
        for month_0 in range(Utils.MONTHS_IN_YEAR):
            self.step_month(year_0, month_0)

        for species in self.species:
            species.population = np.insert(species.population, 0, 0)

    def step_month(self, year_0: int, month_0: int) -> None:
        month = month_0 + 1
        if month in self.environment.plant_growth_months:
            self.reset_food_layers()

        if self.total_water_curr + self.environment.water_replenish < self.total_water_init:
            self.total_water_curr += self.environment.water_replenish
        else:
            self.reset_water_curr()

        for species in self.species:
            species.update_current_fertile_season(month)

        for _ in range(Utils.DAYS_IN_MONTH):
            self.step_day(month)

        if self.do_render:
            self.render(year_0, month_0)

    def step_day(self, month: int) -> None:
        # TODO erst updates berechnen und dann zusammen ausführen, um zu verhindern dass bunnies fressen und der rest hungert
        for species in self.species:
            psum = species.psum()

            species.population[0] *= 1 - Utils.convert_probs(species.infant_mortality)

            food_consumption = species.food_consumption * psum
            available_food = self.food_for_species(species)

            if species.current_fertile_season is not None and available_food > food_consumption:
                gauss_factor = \
                    Utils.gaussian_splits(len(species.current_fertile_season)) \
                    [species.current_fertile_season.index(month)]

                species.do_segs(
                    self.environment.fertility,
                    gauss_factor,
                    available_food,
                    self.environment.plant_growth_months,
                    month)
                assert psum >= 0

            # andrew
            if species.water_consumption * psum > self.environment.water_storage:
                species.verdursten_time_curr -= 1
            else:
                species.reset_verdursten()

            # ashley
            if food_consumption > available_food:
                species.starvation_time_curr -=1
            else:
                species.reset_starvation()

            for f in \
            [ species.do_starve(available_food)
            , species.do_thirst(self.environment.water_storage)
            , species.do_widespread_industrial_sabotage_uwu(self.environment.environment_deaths) ]:
                f()
                psum = species.psum()
                assert psum >= 0

            species.population_hist = np.append(species.population_hist, psum)
            species.population = Utils.do_death(species.population, species.age_death, species.max_age)

            # TODO only if EVERY species is ded
            # if species.psum() == 0:
            #     raise Simulation.AllDead()

            reduction = food_consumption / len(species.food_sources)
            groups = self.food_groups_for_species(species)

            # "katzen hochgewürgte haarball kotze"
            for (g, v) in { g: (
                tmp := v - reduction,
                reduction := reduction + (0 if tmp >= 0 else -tmp / r),
                0 if tmp < 0 else tmp)[-1]
              for ((g, v), r) in zip(
                    groups.items(),
                    list(range(len(groups) - 1, 0, -1)) + [1])}.items():
                self.food_layers_curr.__dict__[g.value] = v

################################################################################

environment = Environment(
    biotope_type_dist    = {Biotope.Mischwald: 0.65, Biotope.Wiese: 0.35},
    environment_deaths   = 0.005,
    fertility            = 0.999,
    minimum_food         = 1000,
    plant_growth_months  = [3, 4, 5, 6, 7, 8, 9, 10],
    simulation_area      = 10,
    simulation_time      = 10,
    water_storage        = 1_000_000,
    water_replenish      = 100_000,
)

bunnies = Species(
    name = "bunnnygirls",

    age_death         = 12,
    age_mature        = 1,
    fertile_seasons   = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    food_consumption  = 1.35,
    food_sources      = set([BiomassType.Ground, BiomassType.Bushes]),
    infant_mortality  = 0.3,
    manual_dist       = {4: 200, 1: 120},
    # mass_food         = 4.5,
    n_birth           = 3,
    segs_probability  = 0.999,
    starvation_time   = 14,
    verdursten_time   = 14,
    water_consumption = 0,
    water_sources     = WaterSource.Implicit,
)

deers = Species(
    name = "senpais",

    age_death         = 14,
    age_mature        = 3,
    fertile_seasons   = [[5, 6, 7]],
    food_consumption  = 4,
    food_sources      = set([BiomassType.Ground, BiomassType.Bushes]),
    infant_mortality  = 0.25,
    manual_dist       = {4: 200, 1: 120},
    # mass_food         = 25,
    n_birth           = 2,
    segs_probability  = 0.9,
    starvation_time   = 7,
    verdursten_time   = 7,
    water_consumption = 0,
    water_sources     = WaterSource.Implicit,
)

cows = Species(
    name = "cowgirls",

    age_death         = 20,
    age_mature        = 1,
    fertile_seasons   = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
    food_consumption  = 60,
    food_sources      = set([BiomassType.Ground]),
    infant_mortality  = 0.3,
    manual_dist       = {4: 50, 1: 10},
    # mass_food         = 900,
    n_birth           = 1,
    segs_probability  = 0.9,
    starvation_time   = 7,
    verdursten_time   = 3,
    water_consumption = 100,
    water_sources     = WaterSource.Explicit,
)

sim = Simulation(environment, [bunnies, deers, cows]) # Rehehe
# sim.do_render = False
print("HERE GO HERE PLEASE =============================")
sim.run()
input("enter to stop")
# plt.ioff()
# plt.show() # type: ignore

# we stay silly :3
