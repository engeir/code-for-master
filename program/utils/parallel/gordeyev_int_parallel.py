"""Implementation of parallel computation of
the Gordeyev integral as a function of frequency.
"""

import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.special as sps
import scipy.constants as const
import scipy.integrate as si

from inputs import config as cf


def integrate(m, T, nu, y, function, kappa=None):
    """Integrate from `0` to `Y_MAX` with an integrand on the form
    `e^{-iwy}f(y)`, for every value in the np.ndarray `w`.

    Arguments:
        m {float} -- mass [kg]
        T {float} -- temperature [K]
        nu {float} -- collision frequency [Hz]
        y {np.ndarray} -- integration sample points
        function {class object} -- object from an integrand class

    Keyword Arguments:
        kappa {int or float} -- index determining the order of the
            kappa VDFs (default: {None})

    Returns:
        np.ndarray -- a scaled version of the result from the
            integration based on Hagfors [1968]
    """
    idx = set(enumerate(cf.w))
    f = function.integrand()
    func = partial(parallel, y, f)
    pool = mp.Pool()
    pool.map(func, idx)
    pool.close()
    if function.the_type == 'kappa':  # $\label{lst:gordeyev_scale}$
        a = array / (2**(kappa - 1 / 2) * sps.gamma(kappa + 1 / 2))
    elif function.the_type == 'a_vdf':
        # Characteristic velocity scaling
        a = 4 * np.pi * T * const.k * array / m * function.char_vel
    else:
        a = array
    if function.the_type == 'a_vdf':
        F = a
    else:
        F = 1 - (1j * cf.w + nu) * a
    return F


def parallel(y, f, index):
    array[index[0]] = simpson(index[1], y, f)


def simpson(w, y, f):
    val = np.exp(- 1j * w * y) * f

    sint = si.simps(val, y)
    return sint


def shared_array(shape):
    """
    Form a shared memory numpy array.

    https://tinyurl.com/c9m75k2
    """

    shared_array_base = mp.Array(ctypes.c_double, 2 * shape[0])
    shared_arr = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_arr = shared_arr.view(np.complex128).reshape(*shape)
    return shared_arr


array = shared_array((int(cf.F_N_POINTS),))
