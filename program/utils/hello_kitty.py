"""Script for calculating the peak power of the plasma line at different pitch angles, height and ToD.
"""

import numpy as np  # pylint: disable=C0413
import matplotlib  # pylint: disable=C0413
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.integrate as si

# from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    # 'font.family': 'Ovo',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class HelloKitty:
    def __init__(self):
        # self.Z = np.arange(100, 350, 50)
        self.Z = np.linspace(4e11, 14e11, 10)
        self.A = 44 - 20 * np.cos(np.linspace(0, np.pi, int(1e1)))
        # print(len(self.Z) * len(self.A))
        self.g = np.zeros((len(self.Z), len(self.A)))
        self.create_data()
        self.plot_data()

    def create_data(self):
        sys_set = {'B': 5e-4, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'gauss_shell', 'area': False}
        for i, z in enumerate(self.Z):
            # plt.figure()
            # sys_set['Z'] = z
            sys_set['NE'] = z
            for j, a in enumerate(self.A):
                sys_set['THETA'] = a * np.pi / 180
                f, s, _ = isr.isr_spectrum('a_vdf', sys_set, **params)
                # plt.plot(f, s)
                res = si.simps(s, f)
                print(f'{res:.4e}')
                # s = np.random.uniform(0, 200)
                self.g[i, j] = res
            # plt.show()

    def plot_data(self):
        # Hello kitty figure
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
        plt.savefig('hello_kitty.pdf', bbox_inches='tight', dpi=200)
        plt.savefig('hello_kitty.pgf', bbox_inches='tight')
        
        # Plot of each angle
        plt.figure()
        for i in range(self.g.shape[1]):
            plt.plot(self.g[:, i])
        plt.show()
