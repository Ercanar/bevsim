import numpy as np

# gaussian split distribution ratios
gaussian_splits = [
    np.array([0]),
    np.array([100]) / 100,
    np.array([50  , 50]) / 100,
    np.array([9.1 , 81.8, 9.1]) / 100,
    np.array([2.3 , 47.7, 47.7, 2.3]) / 100,
    np.array([0.8 , 20.4, 57.6, 20.4, 0.8]) / 100,
    np.array([0.4 , 8.7 , 40.9, 40.9, 8.7 , 0.4]) / 100,
    np.array([0.2 , 4.1 , 24.1, 43.2, 24.1, 4.1 , 0.2]) / 100,
    np.array([0.1 , 2.1 , 13.6, 34.1, 34.1, 13.6, 2.1 , 0.1]) / 100,
    np.array([0.15, 1.2 , 7.8 , 23.7, 34.3, 23.7, 7.8 , 1.2 , 0.15]) / 100,
    np.array([0   , 0.8 , 4.7 , 15.7, 28.8, 28.8, 15.7, 4.7 , 0.8, 0]) / 100,
    np.array([0.1 , 0.5 , 2.9 , 10.3, 22  , 28.4, 22  , 10.3, 2.9, 0.5, 0.1]) / 100,
    np.array([0.1 , 0.3 , 1.9 , 6.8 , 16.1, 24.8, 24.8, 16.1, 6.8, 1.9, 0.3, 0.1]) / 100,
]

# biotop types into three types of plant biomass (ground, bushes, treetops) per 1 km^3 per year in kg
plant_biomass_calc = [
        ("mischwald",     [500000 , 250000, 10000000]),
        ("nadelwald",     [500000 , 250000, 20000000]),
        ("laubwald",      [500000 , 250000, 15000000]),
        ("wiese",         [1000000, 0     , 0       ]),
        #("fluss",        [       ,       ,         ]),
        #("strand",       [       ,       ,         ]),
        #("sumpf",        [       ,       ,         ]),
        #("gewaesser",    [       ,       ,         ]),
]
