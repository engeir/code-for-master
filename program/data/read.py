import os
import sys

import ast
import numpy as np
from scipy.io import loadmat
import scipy.constants as const
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

if __name__ != '__main__':
    from inputs import config as cf

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


def f_0_maxwell(v, params):
    # NOTE: Normalized to 1D
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 1 / 2)
    func = A * np.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def interpolate_data(v, params):
    if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
        f_1 = np.linspace(1, 600, 600)
        energies = np.linspace(1, 110, 600)  # electronvolt
    else:
        if __name__ == '__main__':
            path = 'Arecibo-photo-electrons/'
        else:
            # path = 'data/Arecibo-photo-electrons/'
            path = 'data/arecibo2/'
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

def plot_interp(v, params):
    if __name__ == '__main__':
        # path = 'Arecibo-photo-electrons/'
        path = 'arecibo2/'
    else:
        # path = 'data/Arecibo-photo-electrons/'
        path = 'data/arecibo2/'
    x = loadmat(path + params['mat_file'])
    data = x['fe_zmuE']
    # sum_over_pitch = data[:, :10, :]  # removes j-dimansion through dot-product
    sum_over_pitch = np.einsum('ijk->ik', data) / 18  # removes j-dimansion through dot-product
    idx = int(np.argwhere(read_dat_file('z4fe.dat')==params['Z']))
    f_1 = sum_over_pitch[idx, :]
    energies = read_dat_file('E4fe.dat')

    velocities = (2 * energies * const.eV / params['m'])**.5
    new_f1 = np.interp(v, velocities, f_1)
    f_0 = f_0_maxwell(v, params)
    f0_f1 = f_0 + new_f1
    new_f0f1 = np.maximum(f_0, new_f1)

    # Scale v to get energy instead
    v = 1 / 2 * const.m_e * v**2 / const.eV
    velocities = energies

    plt.figure(figsize=(6, 3))
    plt.semilogy(v, f0_f1, '-')
    plt.semilogy(v, new_f0f1, '--')
    plt.semilogy(v, f_0, '-.')
    plt.semilogy(velocities, f_1, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    plt.legend([r'$f_{0,\mathrm{M}} + f_{0,\mathrm{S}}$',
                'np.maximum(' + r'$f_{0,\mathrm{M}}, f_{0,\mathrm{S}}$' + ')',
                r'$f_{0,\mathrm{M}}$',
                r'$f_{0,\mathrm{S}}$'])
    # plt.xlim([5.8e5, 8e5])
    plt.xlim([1.2, 1.9])
    # plt.ylim([5e-13, 3e-11])
    plt.ylim([5e-14, 3e-12])
    plt.xlabel(r'Energy, $E$ [eV]')
    plt.ylabel('VDF, ' + r'$f_0$')
    # plt.savefig('../../../../report/master-thesis/figures/' + \
    #             'interp_real_data_energyscale.pgf', bbox_inches='tight')
    # plt.savefig(f'../../figures/interp_real_data.pgf', bbox_inches='tight')
    plt.show()

def view_mat_file():
    # path = 'Arecibo-photo-electrons/'
    path = 'arecibo2/'
    x = loadmat(path + 'fe_zmuE-07.mat')
    data = x['fe_zmuE']
    data = data[:, :10, :]
    data = np.einsum('ijk->ik', data) / 10
    idx = int(np.argwhere(read_dat_file('z4fe.dat') == 599))
    # idx = int(np.argwhere(read_dat_file('z4fe.dat') == 300))
    data = data[idx, :]
    E = np.linspace(1, 110, len(data))

    _, ax = plt.subplots(figsize=(6, 3))
    ax.plot(E, data.T, 'g')
    plt.xlabel(r'Energy, $E$ [eV]')
    plt.ylabel('VDF, ' + r'$f_{0,\mathrm{S}}$')
    axins = inset_axes(ax, 3.1, 1.5, loc='upper right')
    axins.plot(E, data.T, 'g')
    x1, x2, y1, y2 = 13, 27, 1e-15, 5.57e-14
    # x1, x2, y1, y2 = 18, 30, 0, 5e-13
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    x00, x01, x10, x11 = 17.8, 19.2, 23.3, 24.7
    # x00, x01, x10, x11, x20, x21 = 21.7, 22.3, 23.5, 24.1, 26.5, 27.2
    axins.axvspan(x00, x01, alpha=0.5, color='g')
    axins.axvspan(x10, x11, alpha=0.5, color='g')
    # axins.axvspan(x20, x21, alpha=0.5, color='g')
    y = 5e-15
    # y = 4e-13
    axins.text(18.5, y, '1', fontsize=13, horizontalalignment='center')
    axins.text(24, y, '2', fontsize=13, horizontalalignment='center')
    # axins.text(26.8, y, '3', fontsize=13, horizontalalignment='center')
    # plt.yticks(visible=False)
    plt.gca().axes.yaxis.set_ticklabels([])
    # axins.ticklabel_format(useOffset=False)
    # plt.tick_params(axis='y', which='both', left=False,
    #                 right=False, labelleft=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # plt.savefig('../../../../report/master-thesis/figures/' + \
    #             'in_use/hello_kitty_1.pgf', bbox_inches='tight')
    plt.show()

def read_dat_file(file):
    """Return the contents of a .dat file as a single numpy row vector.

    Arguments:
        file {str} -- the file name of the .dat file

    Returns:
        np.ndarray -- contents of the .dat file
    """
    l = np.array([])
    if __name__ == '__main__':
        path = 'Arecibo-photo-electrons/'
    else:
        path = 'data/Arecibo-photo-electrons/'
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

if __name__ == '__main__':
    # x = np.linspace(0, 6e6, 1000)
    # param = {'T': 2000, 'm': const.m_e, 'mat_file': 'fe_zmuE-07.mat', 'Z': 599}
    # plot_interp(x, param)
    # interpolate_data(x, param)
    # theta_lims, E4fe, SzeN, timeOfDayUT, z4fe
    # Arecibo is 4 hours behind UT, [9, 16] UT = [5, 12] local time
    # x = loadmat('Arecibo-photo-electrons/' + 'fe_zmuE-15.mat')
    # data = x['fe_zmuE']
    view_mat_file()
    # dat_file = read_dat_file('SzeN.dat')
    # print(dat_file)
