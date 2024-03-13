#!/usr/bin/env python3
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from preset1 import *
from lookup import *
from environment1 import *
from time import time
from myTypes import *

# TODO funktion fuer simulation, inputs: array von liste an presets, available food, etc

plt.plot(range(len(bev_hist)), bev_hist)
plt.xlabel("time in days", fontsize="xx-large")
plt.ylabel("bev in #", fontsize="xx-large")
plt.grid()
plt.show()
