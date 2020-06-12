import os
import sys

import ast
import numpy as np
from scipy.io import loadmat
import scipy.constants as const
import matplotlib.pyplot as plt


def f_0_maxwell(v, params):
    # NOTE: Normalized to 1D
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 1 / 2)
    func = A * np.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def interpolate_data(v, params):
    if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
        # f_1 = np.linspace(1, 600, 600)
        # energies = np.linspace(1, 110, 600)  # electronvolt
        path = 'data/arecibo/'
        if not os.path.exists(path):
            path = 'program/data/arecibo/'
        x = loadmat(path + params['mat_file'])
        data = x['fe_zmuE']
        if isinstance(params['pitch_angle'], list):
            if all(isinstance(x, int) for x in params['pitch_angle']):
                sum_over_pitch = data[:, params['pitch_angle'], :]
                norm = len(params['pitch_angle'])
        else:
            norm = 18
        sum_over_pitch = np.einsum('ijk->ik', data) / norm  # removes j-dimansion through dot-product
        idx = int(np.argwhere(read_dat_file('z4fe.dat')==params['Z']))
        f_1 = sum_over_pitch[idx, :]
        energies = read_dat_file('E4fe.dat')
    else:
        path = 'data/arecibo/'
        x = loadmat(path + params['mat_file'])
        data = x['fe_zmuE']
        if isinstance(params['pitch_angle'], list):
            if all(isinstance(x, int) for x in params['pitch_angle']):
                sum_over_pitch = data[:, params['pitch_angle'], :]
                norm = len(params['pitch_angle'])
        else:
            norm = 18
        sum_over_pitch = np.einsum('ijk->ik', data) / norm  # removes j-dimansion through dot-product
        idx = int(np.argwhere(read_dat_file('z4fe.dat')==params['Z']))
        f_1 = sum_over_pitch[idx, :]
        energies = read_dat_file('E4fe.dat')

    velocities = (2 * energies * const.eV / params['m'])**.5
    new_f1 = np.interp(v, velocities, f_1)
    f_0 = f_0_maxwell(v, params)
    f0_f1 = f_0 + new_f1

    return f0_f1


def read_dat_file(file):
    """Return the contents of a .dat file as a single numpy row vector.

    Arguments:
        file {str} -- the file name of the .dat file

    Returns:
        np.ndarray -- contents of the .dat file
    """
    l = np.array([])
    path = 'data/arecibo/'
    if not os.path.exists(path):
        path = 'program/data/arecibo/'
    with open(path + file) as f:
        ll = f.readlines()
        ll = [x.strip() for x in ll]
        l = np.r_[l, ll]
    if len(l) == 1:
        for p in l:
            l = p.split()
    e = []
    for p in l:
        k = ast.literal_eval(p)
        e.append(k)
    return np.array(e)
