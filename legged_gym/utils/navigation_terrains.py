# Define some special terrains for navigation tasks

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt

