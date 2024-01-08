import numpy as np

initial_distribution = np.array([0, 0, 0, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # age distribution of starting population
species_factor       = 0.999 # probability of segs in season
age_mature           = 4 # age of seggsual maturity
age_death            = 16 # life expectency, gauss distribution with sigma = lifespan/16 (here sigma=1)
n_birth              = 2 # erwartungs anzahl kids pro birth
fertile_months       = [[3, 4, 5], [6, 7, 8]] # months mit segs, seasons possible, cannot overlap
infant_mortality     = 10 # promille of infants that die in first year
food_consumption     = 15 # food consumption per capita per month in kg
# food_sources         = set([FoodSource.Ground, ]) # can be (ground, bushes, treetops) or other animals or combinations
mass_food            = 4 # mass of specimen in kg for carnivore food calculation
starvation_time      = 3 # average days to starve
water_sources        = "implicit" # implicit or explicit, ersteres ist im food/luft/tau etc enthalten, zweiteres muss extra gefunden werden (pfuetzen, seen, fluesse etc)
