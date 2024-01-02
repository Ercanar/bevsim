import numpy as np

initial = 30 # starting population
initial_distribution = [0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # age distribution of starting population
species_factor = 0.9 # probability of segs in season
age_mature = 4 # age of sexual maturity
age_death = 16 # life expectency, gauss distribution with sigma = lifespan/16 (here sigma=1)
n_birth = 1 # erwartungs anzahl kids pro birth
food = 100 # anzahl sustainable tiere
fertile_months = [[3, 4, 5], [6, 7, 8]] # months mit segs, seasons possible, cannot overlap
