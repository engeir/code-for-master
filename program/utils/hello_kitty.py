"""Script for calculating the peak power of the plasma line
at different pitch angles, height and time of day.
"""

import sys, os
import time
import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import scipy.integrate as si
import scipy.constants as const
import scipy.signal as signal
from lmfit.models import LorentzianModel
from tqdm import tqdm

from utils import spectrum_calculation as isr
from inputs import config as cf

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class HelloKitty:
    def __init__(self):
        # For plot nr. 1, set 'self.vol = 1. For plot nr. 2, set self.vol = 2.
        self.vol = 1
        if self.vol == 1:
            self.Z = np.linspace(1e11, 8e11, 60)
        else:
            self.Z = np.linspace(2e11, 1e12, 60)
        self.A = 45 + 15 * np.cos(np.linspace(0, np.pi, 30))
        self.g = np.zeros((len(self.Z), len(self.A)))
        self.dots = [[], [], []]
        self.meta = []
        self.F0 = 430e6
        self.K_RADAR = - 2 * self.F0 * 2 * np.pi / const.c  # Radar wavenumber
        save = input('Press "y/yes" to save plot, ' + \
                     'any other key to dismiss.\t').lower()
        if save in ['y', 'yes']:
            self.save = True
        else:
            self.save = False

    def create_data(self):
        # In config, set 'F_MIN': 2.5e6, 'F_MAX': 9.5e6
        # Also, using F_N_POINTS = 1e4 is sufficient.
        if self.vol == 1:
            sys_set = {'K_RADAR': self.K_RADAR, 'B': 35000e-9, 'MI': 16,
                       'NE': 2e10, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000,
                       'T_I': 1500, 'T_ES': 90000,
                       'THETA': 60 * np.pi / 180, 'Z': 599,
                       'mat_file': 'fe_zmuE-07.mat',
                       'pitch_angle': list(range(10))}
        else:
            sys_set = {'K_RADAR': self.K_RADAR, 'B': 35000e-9, 'MI': 16,
                       'NE': 2e10, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000,
                       'T_I': 1500, 'T_ES': 90000,
                       'THETA': 60 * np.pi / 180, 'Z': 300,
                       'mat_file': 'fe_zmuE-07.mat',
                       'pitch_angle': 'all'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        with tqdm(total=len(self.Z) * len(self.A)) as pbar:
            for i, z in enumerate(self.Z):
                sys_set['NE'] = z
                plasma_freq = (sys_set['NE'] * const.elementary_charge**2 /
                               (const.m_e * const.epsilon_0))**.5 / (2 * np.pi)
                cf.I_P['F_MIN'] = plasma_freq
                cf.I_P['F_MAX'] = plasma_freq + 4e5
                cf.f = np.linspace(cf.I_P['F_MIN'], cf.I_P['F_MAX'], int(cf.F_N_POINTS))
                cf.w = 2 * np.pi * cf.f  # Angular frequency
                for j, a in enumerate(self.A):
                    sys_set['THETA'] = a * np.pi / 180
                    old_stdout = sys.stdout
                    f = open(os.devnull, 'w')
                    sys.stdout = f
                    f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
                    sys.stdout = old_stdout
                    plasma_power, energy_interval = self.check_energy(f, s, a)
                    if energy_interval != 0:
                        self.dots[0].append(energy_interval)
                        self.dots[1].append(j)
                        self.dots[2].append(z)
                    self.g[i, j] = plasma_power
                    pbar.update(1)
        self.meta.append(meta_data)

    def check_energy(self, f, s, deg):
        p = int(np.argwhere(s==np.max(s)))
        freq = f[p]
        f_mask = (freq - 5e2 < f) & (f < freq + 5e2)
        x = f[f_mask]
        y = s[f_mask]
        mod = LorentzianModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)
        power = si.simps(out.best_fit, x)

        l = const.c / self.F0
        # Calculate corresponding energy with formula: $ E = 0.5 m_{\mathrm{e}} [f_{\mathrm{r}} \lambda_\Re / (2 \cos\theta)]^2 $
        E_plasma = .5 * const.m_e * (freq * l / (2 * np.cos(deg * np.pi / 180)))**2 / const.eV
        res = 0
        if self.vol == 1:
            if bool(15.58 < E_plasma < 18.42):
                res = 1
            elif bool(22.47 < E_plasma < 23.75):
                res = 2
        else:
            if bool(20.29 < E_plasma < 22.05):
                res = 1
            elif bool(22.45 < E_plasma < 23.87):
                res = 2
            elif bool(25.38 < E_plasma < 27.14):
                res = 3
        return power, res

    def plot_data(self):
        # Hello kitty figure duplication
        self.g = np.c_[self.g, self.g[:, ::-1], self.g, self.g[:, ::-1]]
        self.A = np.r_[self.A, self.A[::-1], self.A, self.A[::-1]]
        dots_x = []
        dots_y = []
        for i, d in enumerate(self.dots[1]):
            arg = np.argwhere(self.A == self.A[d])
            dots_x = np.r_[dots_x, arg[:2, 0]]
            dots_y = np.r_[dots_y, np.ones(len(arg[:2, 0])) * self.dots[2][i]]

        f = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(self.g, extent=[0, len(self.A) - 1,
                                        np.min(self.Z), np.max(self.Z)],
                        origin='lower', aspect='auto', cmap='gist_heat')
        plt.scatter(dots_x, dots_y, s=3)
        plt.ylabel(r'Electron number density, $n_{\mathrm{e}}$')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        ax1 = plt.subplot(gs[1])
        ax1.plot(self.A)
        plt.xlim([0, len(self.A) - 1])
        plt.yticks([30, 45, 60])
        plt.ylabel('Aspect angle')
        axs = []
        axs += [ax0]
        axs += [ax1]
        gs.update(hspace=0.05)
        f.colorbar(im, ax=axs).ax.set_ylabel('Echo power')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)

        if self.save:
            save_path = '../../../report/master-thesis/figures'
            if not os.path.exists(save_path):
                save_path = '../figures'
                os.makedirs(save_path, exist_ok=True)
            tt = time.localtime()
            the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
            save_path = f'{save_path}/hello_kitty_{the_time}'
            self.meta.insert(0, {'F_MAX': cf.I_P['F_MAX'], 'V_MAX': cf.V_MAX,
                                 'F_N_POINTS': cf.F_N_POINTS, 'Y_N_POINTS': cf.Y_N_POINTS,
                                 'V_N_POINTS': cf.V_N_POINTS})

            pdffig = PdfPages(str(save_path) + '.pdf')
            metadata = pdffig.infodict()
            metadata['Title'] = f'Hello Kitty plot'
            metadata['Author'] = 'Eirik R. Enger'
            metadata['Subject'] = f"Plasma line power as a function of ' + \
                                  'electron number density and aspect angle."
            metadata['Keywords'] = f'{self.meta}'
            metadata['ModDate'] = datetime.datetime.today()
            pdffig.attach_note('max(s), 100percent power')
            plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
            pdffig.close()
            plt.savefig(f'{save_path}.pgf', bbox_inches='tight')
            np.savez(f'{save_path}', angle=self.A, density=self.Z, power=self.g, dots=self.dots)

        plt.show()

    def run(self):
        self.create_data()
        self.plot_data()