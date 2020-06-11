"""Constants used system wide.
"""

import os
import sys

import numpy as np


# Check if a test is running. Potential paths are
# ['pytest.py', 'pytest', 'test_ISR.py', '__main__.py', 'python3.7 -m unittest']
# or check if 'main.py' was used.
if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
    # DO NOT EDIT
    F_N_POINTS = 1e1
    Y_N_POINTS = 1e1
    V_N_POINTS = 1e1
else:
    F_N_POINTS = 1e3  # Number of sample points in frequency
    Y_N_POINTS = 8e5  # Number of sample points in integral variable
    V_N_POINTS = 4e4  # Number of sample points in velocity integral variable
# Adds one sample to get an even number of bins, which in
# turn give better precision in the Simpson integration.
Y_N_POINTS += 1
V_N_POINTS += 1
Y_MAX_e = 1.5e-4  # Upper limit of integration (= infinity)
Y_MAX_i = 1.5e-2
# Based on E = 110 eV -> 6.22e6 m/s
V_MAX = 6e6
ORDER = 3

I_P = {'F_MIN': - 2e6, 'F_MAX': 12e6}
f = np.linspace(I_P['F_MIN'], I_P['F_MAX'], int(F_N_POINTS))
f = (f / I_P['F_MAX'])**3 * I_P['F_MAX']
w = 2 * np.pi * f  # Angular frequency
