import numpy as np

# food = 2000 # plant biomass
biotop_area = 1  # in km^3
biotop_type_dist = {
    "mischwald": 0.65,
    "wiese": 0.35,
}  # contens of biotop with distribution
plant_growth_months = [4, 5, 6, 7, 8, 9]  # months where biomass is replenished
fertility = 1 - (0.001)  # fertility penalty for specific environments
environment_deaths = 10  # death per 1000 people per month, maximum 999 for extinktion
water_storage = 10000 # reservoir of water in biotop in l (filled up by rain or river)
