"""Script for calculating the peak power of the plasma line at different pitch angles, height and ToD.
"""

import sys, os
from contextlib import contextmanager

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.integrate as si
from tqdm import tqdm

# from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    # 'font.family': 'Ovo',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})

# @contextmanager
# def suppress_stdout():
#     with open(os.devnull, "w") as devnull:
#         old_stdout = sys.stdout
#         sys.stdout = devnull
#         try:
#             yield
#         finally:
#             sys.stdout = old_stdout


class HelloKitty:
    def __init__(self):
        # self.Z = np.arange(100, 350, 50)
        self.Z = np.linspace(2e10, 6e11, 70)
        self.A = 45 + 15 * np.cos(np.linspace(0, np.pi, 30))  # 25))
        # print(len(self.Z) * len(self.A))
        self.g = np.zeros((len(self.Z), len(self.A)))
        self.create_data()
        self.plot_data()

    def create_data(self):
        # Seems to be close to working with gauss_shell, f_0 = 933e6, NE ≈ [1e11, 2e12], 1e3, 5e4, 2e4
        # sys_set = {'B': 35000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 100, 'NU_I': 0, 'T_E': 5000, 'T_I': 1000, 'T_ES': 90000,
        #            'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        # params = {'kappa': 8, 'vdf': 'gauss_shell', 'area': False}
        # Seems to be close to working with real_data, f_0 = 430e6, NE = [2e10, 6e11], 1e4, 4e5, 1e4
        sys_set = {'B': 35000e-9, 'MI': 16, 'NE': 2e10, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000, 'T_I': 1500, 'T_ES': 90000,
                   'THETA': 60 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        # c = 0
        with tqdm(total=len(self.Z) * len(self.A)) as pbar:
            for i, z in enumerate(self.Z):
                # plt.figure()
                # sys_set['Z'] = z
                sys_set['NE'] = z
                for j, a in enumerate(self.A):
                    sys_set['THETA'] = a * np.pi / 180
                    old_stdout = sys.stdout
                    f = open(os.devnull, 'w')
                    sys.stdout = f
                    f, s, _ = isr.isr_spectrum('a_vdf', sys_set, **params)
                    sys.stdout = old_stdout
                    # plt.plot(f, s)
                    res = si.simps(s, f)
                    # print(f'{res:.4e}')
                    # s = np.random.uniform(0, 200)
                    self.g[i, j] = res
                    # self.g[i, j] = np.max(s)
                    pbar.update(1)
                    # c += 1
                    # print(f'{c}/{len(self.Z) * len(self.A)}', end='\r')
            # plt.show()

    def plot_data(self):
        # Hello kitty figure duplication
        self.g = np.c_[self.g, self.g[:, ::-1], self.g, self.g[:, ::-1]]
        self.A = np.r_[self.A, self.A[::-1], self.A, self.A[::-1]]
        # Z, A = np.meshgrid(self.Z, self.A)
        f = plt.figure(figsize=(6, 6))
        # f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(self.g, extent=[20, 60, np.min(self.Z), np.max(self.Z)],
                        origin='lower', aspect='auto', cmap='gist_heat')
        plt.ylabel('Height')
        # im = ax1.imshow(self.g, extent=[20, 60, 100, 350], origin='lower', aspect='auto')
        # plt.colorbar().ax.set_ylabel('Echo Power')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        ax1 = plt.subplot(gs[1])
        ax1.plot(self.A)
        plt.xlim([0, len(self.A) - 1])
        plt.ylabel('Pitch angle')
        axs = []
        axs += [ax0]
        axs += [ax1]
        gs.update(hspace=0.05)
        f.colorbar(im, ax=axs).ax.set_ylabel('Echo Power')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        plt.savefig('hello_kitty_big_one.pdf', bbox_inches='tight', dpi=200)
        plt.savefig('hello_kitty_big_one.pgf', bbox_inches='tight')

        # Plot of each angle
        # plt.figure()
        # for i in range(self.g.shape[1]):
        #     plt.plot(self.g[:, i])
        plt.show()
