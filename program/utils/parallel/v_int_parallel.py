"""Implementation of parallel computation of the velocity integrals for the integral variable y.
"""

import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.integrate as si
from tqdm import tqdm

from inputs import config as cf


def integrand(y, params, v, f):
    """Integrate from 0 to V_MAX with an integrand on the form e^{-iwt}f(t),
    for every value in the np.ndarray w.

    Arguments:
        y {np.ndarray} -- sample points of integration variable
        params {dict} -- plasma parameters
        v {np.ndarray} -- sample points of VDF
        f {np.ndarray} -- value of VDF at sample points

    Returns:
        np.ndarray -- the value of the velocity integral at every sample of the integration variable
    """
    idx = set(enumerate(y))
    func = partial(parallel, params, v, f)
    pool = mp.Pool(processes=96)
    # tqdm give a neat progress bar for the iterative process
    # with tqdm(total=len(y)) as pbar:
    #     for _ in pool.map(func, idx):
    #         pass
    pool.map(func, idx)
            # pbar.set_description("Calculating velocity integral")
            # pbar.update(1)
    pool.close()
    return array


def parallel(params, v, f, index):
    array[index[0]] = v_int_integrand(index[1], params, v, f)


def v_int_integrand(y, params, v, f):
    sin = np.sin(p(y, params) * v)
    val = v * sin * f
    res = si.simps(val, v)
    return res


def p(y, params):
    k_perp = cf.K_RADAR * np.sin(params['THETA'])
    k_par = cf.K_RADAR * np.cos(params['THETA'])
    return (2 * k_perp**2 / params['w_c']**2 * (1 - np.cos(y * params['w_c'])) + k_par**2 * y**2)**.5


def shared_array(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """

    shared_array_base = mp.Array(ctypes.c_double, shape[0])
    shared_arr = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_arr = shared_arr.view(np.double).reshape(*shape)
    return shared_arr


array = shared_array((int(cf.Y_N_POINTS),))
