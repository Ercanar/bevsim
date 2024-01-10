from dataclasses import dataclass
from enum import Enum

fields = ["Ground", "Bushes", "Treetop"]
BiomassType = Enum("BiomassType", [[f, f] for f in fields])
Biomass = dataclass(type("Biomass", (object,), {
    "__call__": (lambda self, val: self.__dict__[val.value]),
    "__annotations__": {field: int for field in fields}}))

WaterSource = Enum("WaterSource", ["Implicit", "Explicit"])

class Memo:
    def __init__(self, calc):
        self.table = {}
        self.calc = calc

    def __call__(self, index):
        if not index in self.table:
            self.table[index] = self.calc(index)

        return self.table[index]
