import numpy as np
from myTypes import Biomass

# biotop types into three types of plant biomass (ground, bushes, treetops) per 1 km^3 per year in kg
# TODO: real values
plant_biomass_calc = {
    "mischwald":     Biomass(  500_000, 250_000, 10_000_000),
    "nadelwald":     Biomass(  500_000, 250_000, 20_000_000),
    "laubwald":      Biomass(  500_000, 250_000, 15_000_000),
    "wiese":         Biomass(1_000_000,       0,          0), # wert ist für trockenes weed
    # "fluss":         Biomass(       ,       ,         ),
    # "strand":        Biomass(       ,       ,         ),
    # "sumpf":         Biomass(       ,       ,         ),
    # "gewässer":      Biomass(       ,       ,         ),
}
