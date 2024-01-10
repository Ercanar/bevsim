from dataclasses import dataclass
from enum import Enum

fields = ["Ground", "Bushes", "Treetop"]
BiomassType = Enum("BiomassType", [[f, f] for f in fields])
Biomass = dataclass(type("Biomass", (object,), {
    "get": (lambda self, val: self.__dict__[val.value]),
    "__annotations__": {field: int for field in fields}}))

WaterSource = Enum("WaterSource", ["Implicit", "Explicit"])
