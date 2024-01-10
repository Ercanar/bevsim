from dataclasses import dataclass
from enum import Enum

fields = ["Ground", "Bushes", "Treetop"]
BiomassType = Enum("BiomassType", fields)
Biomass = dataclass(type("Biomass", (object,), {
    "__annotations__": {field: int for field in fields}}))
