"""Script for calculating the peak power of the plasma line at different pitch angles, height and ToD.
"""

import numpy as np  # pylint: disable=C0413
import matplotlib  # pylint: disable=C0413
import matplotlib.pyplot as plt
from matplotlib import gridspec

# from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'Ovo',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class HelloKitty:
    def __init__(self):
        self.Z = np.arange(100, 350, 50)
        self.A = 40 - 20 * np.cos(np.linspace(0, np.pi, int(1e1)))
        print(len(self.Z) * len(self.A))
        self.g = np.zeros((len(self.Z), len(self.A)))
        self.create_data()
        self.plot_data()

    def create_data(self):
        sys_set = {'B': 5e-4, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        for i, z in enumerate(self.Z):
            sys_set['Z'] = z
            for j, a in enumerate(self.A):
                sys_set['THETA'] = a * np.pi / 180
                _, s, _ = isr.isr_spectrum('a_vdf', sys_set, **params)
                # s = np.random.uniform(0, 200)
                self.g[i, j] = np.max(s)

    def plot_data(self):
        # Z, A = np.meshgrid(self.Z, self.A)
        f = plt.figure(figsize=(6, 6))
        # f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        # plt.contourf(Z, A. self.g)
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
        # ax2.plot(self.A)
        # plt.tight_layout()
        plt.savefig('hello_kitty.pdf', bbox_inches='tight', dpi=200)
        plt.savefig('hello_kitty.pgf', bbox_inches='tight')
        plt.show()
