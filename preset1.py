import numpy as np

initial_distribution = np.array([0, 0, 0, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # age distribution of starting population
species_factor = 0.999 # probability of segs in season
age_mature = 4 # age of seggsual maturity
age_death = 16 # life expectency, gauss distribution with sigma = lifespan/16 (here sigma=1)
n_birth = 5 # erwartungs anzahl kids pro birth
fertile_months = [[3, 4, 5], [6, 7, 8]] # months mit segs, seasons possible, cannot overlap
