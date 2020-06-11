import sys
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as PathEffects
import numpy as np
import scipy.constants as const

from inputs import config as cf

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})

if __name__ != '__main__':
    from utils import spectrum_calculation as isr


class Reproduce(ABC):
    """Abstract base class to reproduce figures.

    Arguments:
        ABC {class} -- abstract base class
    """

    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self, *args, from_file=False):
        if not from_file:
            self.create_from_code()
        else:
            self.create_from_file(*args)

    @abstractmethod
    def create_from_code(self):
        """Method that create needed data.
        """

    def create_from_file(self, *args):
        """Accepts zero, one or two arguments.

        If zero arguments are given, a default path is used to look for files.
        If one argument is given, it should include
            the full path (with or without file ending).
        If two arguments are given, the first should be the path to
            the directory where the file is located, and the second
            argument must be the name of the file.
        """
        if len(args) != 0:
            if len(args) == 1:
                args = args[0]
                parts = args.split('/')
                path = '/'.join(parts[:-1]) + '/'
                name = parts[-1]
            elif len(args) == 2:
                path = args[0]
                name = args[1]
        else:
            path = '../../figures/'
            name = 'hello_kitty_2020_6_9_2--28--4.npz'
        name = name.split('.')[0]
        try:
            f = np.load(path + name + '.npz', allow_pickle=True)
        except Exception:
            sys.exit(print(f'Could not open file {path + name}.npz'))
        sorted(f)
        self.f, self.data, self.meta_data = f['frequency'], list(f['spectra']), list(f['meta'])
        self.legend_txt, self.ridge_txt = list(f['legend_txt']), list(f['ridge_txt'])

        if self.p.save in ['y', 'yes']:
            self.p.save_path = name

    @abstractmethod
    def plot_it(self):
        """Method that plot relevant plots.
        """


class PlotNumerical(Reproduce):
    """Reproduce figure with a comparison between the semi-analytic and numerical implementation.

    In config, set
        'F_MIN': - 2e6, 'F_MAX': 9e6
    Also, using
        F_N_POINTS = 1e3
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 16,
                   'NE': 1e12, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000,
                   'T_I': 1500, 'T_ES': 90000,
                   'THETA': 30 * np.pi / 180, 'Z': 300,
                   'mat_file': 'fe_zmuE-07.mat',
                   'pitch_angle': 'all'}
        params = {'kappa': 3, 'vdf': 'maxwell', 'area': False}

        ridge = []
        self.f, s1, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        ridge.append(s1)
        self.meta_data.append(meta_data)
        _, s2, _ = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s2)
        self.data.append(ridge)

        ridge = []
        params['vdf'] = 'kappa'
        self.f, s1, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        ridge.append(s1)
        self.meta_data.append(meta_data)
        _, s2, _ = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s2)
        self.data.append(ridge)

    def plot_it(self):
        for maxwell, data in enumerate(self.data):
            self.plotter(maxwell, data)

    def plotter(self, maxwell, data):
        s1 = data[0]
        s2 = data[1]
        plot = plt.semilogy
        xlim = [1e3, self.f[-1]]
        d = s1 - s2
        rd = d / s1
        plt.figure(figsize=(8, 5))
        plt.subplot(3, 1, 1)
        if maxwell == 0:
            plt.title('Maxwell')
        else:
            plt.title('Kappa')
        plot(self.f, s1, 'k', label='Semi-analytic (SA)')
        plot(self.f, s2, 'r--', label='Numerical (N)')
        plt.legend()
        # plt.xlim(xlim)
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.subplot(3, 1, 2)
        plt.title('Difference (SA - N)')
        plot(self.f, d, 'k', label='Positive')
        plot(self.f, - d, 'r', label='Negative')
        plt.legend()
        # plt.xlim(xlim)
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.subplot(3, 1, 3)
        plt.title('Difference relative to semi-analytic [(SA - N) / SA]')
        plot(self.f, rd, 'k', label='Positive')
        plot(self.f, - rd, 'r', label='Negative')
        plt.legend()
        # plt.xlim(xlim)
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.yticks([1e-9, 1e-6, 1e-3, 1e0])

        plt.tight_layout()

        if self.p.save in ['y', 'yes']:
            self.p.pdffig.attach_note('numerical precision test')
            plt.savefig(self.p.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.p.save_path) + f'_page_{self.p.page}.pgf', bbox_inches='tight')
            self.p.page += 1


class PlotTestDebye(Reproduce):
    """Reproduce figure of IS spectra using two kappa
    dist with and without Debye length correction.

    In config, set
        'F_MIN': - 2e6, 'F_MAX': 2e6
    Also, using
        F_N_POINTS = 5e5
    is sufficient.
    """

    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        self.legend_txt = [r'$\lambda_{\mathrm{D}} = \lambda_{\mathrm{D},\kappa}$', r'$\lambda_{\mathrm{D}} = \lambda_{\mathrm{D,M}}$']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 3, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)
        params['debye'] = 'maxwell'
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotMaxwell(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': - 2e6, 'F_MAX': 2e6
    Also, using
        F_N_POINTS = 5e5
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        self.legend_txt = ['Maxwellian']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotKappa(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': - 2e6, 'F_MAX': 2e6
    Also, using
        F_N_POINTS = 5e5
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # Change the value of kappa in params (and in legend_txt) to obtain plots of different kappa value.
        self.legend_txt = [r'$\kappa = 20$']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotSpectra(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': - 2e6, 'F_MAX': 2e6
    Also, using
        F_N_POINTS = 1e5
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        self.legend_txt = ['Maxwellian', r'$\kappa = 20$', r'$\kappa = 8$', r'$\kappa = 3$']
        kappa = [20, 8, 3]
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotIonLine(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': - 3e3, 'F_MAX': 3e3
    Also, using
        F_N_POINTS = 1e3
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c
        self.legend_txt = ['Maxwellian', r'$\kappa = 20$', r'$\kappa = 8$', r'$\kappa = 3$']
        kappa = [20, 8, 3]
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'plot', self.legend_txt)


class PlotPlasmaLine(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': 3.5e6, 'F_MAX': 7e6
    Also, using
        F_N_POINTS = 1e3
    is sufficient.
    """
    def create_from_code(self):
        F0 = 933e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c
        self.legend_txt = ['Maxwellian', r'$\kappa = 20$', r'$\kappa = 8$', r'$\kappa = 3$']
        kappa = [20, 8, 3]
        sys_set = {'K_RADAR': K_RADAR, 'B': 50000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 0 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'plot', self.legend_txt)


class PlotTemperature(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': 3.5e6, 'F_MAX': 7.5e6
    Also, using
        F_N_POINTS = 5e3
    is sufficient.
    """
    def __init__(self, p):
        super(PlotTemperature, self).__init__(p)
        self.f_list = [[], [], []]

    def create_from_file(self, *args):
        """Accepts zero, one or two arguments.

        If zero arguments are given, a default path is used to look for files.
        If one argument is given, it should include
            the full path (with or without file ending).
        If two arguments are given, the first should be the path to
            the directory where the file is located, and the second
            argument must be the name of the file.
        """
        if len(args) != 0:
            if len(args) == 1:
                args = args[0]
                parts = args.split('/')
                path = '/'.join(parts[:-1]) + '/'
                name = parts[-1]
            elif len(args) == 2:
                path = args[0]
                name = args[1]
        else:
            path = '../../figures/'
            name = 'hello_kitty_2020_6_9_2--28--4.npz'
        name = name.split('.')[0]
        try:
            f = np.load(path + name + '.npz', allow_pickle=True)
        except Exception:
            sys.exit(print(f'Could not open file {path + name}.npz'))
        sorted(f)
        self.f, self.data, self.meta_data = f['frequency'], list(f['spectra']), list(f['meta'])
        self.legend_txt, self.ridge_txt = list(f['legend_txt']), list(f['ridge_txt'])

        for r in self.data:
            peak = int(np.argwhere(r[0] == np.max(r[0])))
            self.f_list[0].append(self.f[peak])
            peak = int(np.argwhere(r[1] == np.max(r[1])))
            self.f_list[1].append(self.f[peak])
            peak = int(np.argwhere(r[2] == np.max(r[2])))
            self.f_list[2].append(self.f[peak])

        if self.p.save in ['y', 'yes']:
            self.p.save_path = name

    def create_from_code(self):
        F0 = 933e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c
        T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        self.ridge_txt = [r'$T_{\mathrm{e}} = %d \mathrm{K}$' % j for j in T]
        self.legend_txt = ['Maxwellian', r'$\kappa = 20$', r'$\kappa = 3$']
        sys_set = {'K_RADAR': K_RADAR, 'B': 50000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 2000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 0 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        kappa = [20, 3]
        for t in T:
            ridge = []
            sys_set['T_E'] = t
            self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
            ridge.append(s)
            for k in kappa:
                params['kappa'] = k
                self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
                ridge.append(s)
            self.data.append(ridge)
        self.meta_data.append(meta_data)

        for r in self.data:
            peak = int(np.argwhere(r[0] == np.max(r[0])))
            self.f_list[0].append(self.f[peak])
            peak = int(np.argwhere(r[1] == np.max(r[1])))
            self.f_list[1].append(self.f[peak])
            peak = int(np.argwhere(r[2] == np.max(r[2])))
            self.f_list[2].append(self.f[peak])

    def plot_it(self):
        self.p.plot_ridge(self.f, self.data, 'plot', self.legend_txt, self.ridge_txt)

        T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        plt.figure(figsize=(6, 3))
        plt.plot(T, self.f_list[0], 'k', label='Maxwellian')
        plt.plot(T, self.f_list[1], 'k--', label=r'$\kappa = 20$')
        plt.plot(T, self.f_list[2], 'k:', label=r'$\kappa = 3$')
        plt.legend()

        if self.p.save in ['y', 'yes']:
            self.p.pdffig.attach_note('freq change')
            plt.savefig(self.p.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.p.save_path) + f'_page_{self.p.page}.pgf', bbox_inches='tight')
            self.p.page += 1


class PlotHKExtremes(Reproduce):
    """Reproduce figure with ridge plot over different temperatures.

    In config, set
        'F_MIN': 2.5e6, 'F_MAX': 9.5e6
    Also, using
        F_N_POINTS = 1e4
    is sufficient.
    """
    def create_from_code(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 16, 'NE': 1e11,
                   'NU_E': 100, 'NU_I': 100, 'T_E': 2000, 'T_I': 1500, 'T_ES': 90000,
                   'THETA': 30 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat',
                   'pitch_angle': list(range(10))}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        # Ridge 1
        ridge = []
        # Line 1
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.meta_data.append(meta_data)
        # Line 2
        sys_set['NE'] = 1e12
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.data.append(ridge)
        self.meta_data.append(meta_data)

        # Ridge 2
        ridge = []
        # Line 1
        sys_set['THETA'] = 60 * np.pi / 180
        sys_set['NE'] = 1e11
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.meta_data.append(meta_data)
        # Line 2
        sys_set['NE'] = 1e12
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.data.append(ridge)
        self.meta_data.append(meta_data)

        self.legend_txt = ['1e11', '1e12']
        self.ridge_txt = ['30', '60']

    def plot_it(self):
        self.p.plot_ridge(self.f, self.data, 'semilogy', self.legend_txt, self.ridge_txt)


class PlotHK:
    """Reproduce the Hello Kitty figures from saved data."""
    def __init__(self, *args):
        """Accepts zero, one or two arguments.

        If zero arguments are given, a default path is used to look for files.
        If one argument is given, it should include
            the full path (with or without file ending).
        If two arguments are given, the first should be the path to
            the directory where the file is located, and the second
            argument must be the name of the file.
        """
        if len(args) != 0:
            if len(args) == 1:
                args = args[0]
                parts = args.split('/')
                path = '/'.join(parts[:-1]) + '/'
                self.name = parts[-1]
            elif len(args) == 2:
                path = args[0]
                self.name = args[1]
        else:
            path = '../../figures/'
            self.name = 'hello_kitty_2020_6_9_2--28--4.npz'
            # self.name = 'hello_kitty_2020_6_8_22--1--51.npz'
        self.name = self.name.split('.')[0]
        try:
            self.file = np.load(path + self.name + '.npz')
        except Exception:
            sys.exit(print(f'Could not open file {path + self.name}'))
        sorted(self.file)

    def shade(self):
        dots_x = []
        dots_y = []
        for i, d in enumerate(self.file['dots'][1]):
            arg = np.argwhere(self.file['angle'] == self.file['angle'][int(d)])
            dots_x = np.r_[dots_x, arg[:1, 0]]
            dots_y = np.r_[dots_y, np.ones(len(arg[:1, 0])) * self.file['dots'][2][i]]

        s = set(self.file['dots'][0])
        for i in s:
            mask = np.argwhere(self.file['dots'][0]==i)
            xs = []
            y_min = []
            y_max = []
            for x in range(30):
                arg = np.argwhere(dots_x[mask].flatten() == x)
                if bool(arg.any()):
                    xs.append(x)
                    y_min.append(np.min(dots_y[mask][arg]))
                    y_max.append(np.max(dots_y[mask][arg]))
            plt.fill_between(xs, y_min, y_max, color='g', alpha=.8)
            x, y = xs[-1], (y_max[-1] + y_min[-1]) / 2
            txt = plt.text(x, y, r'$\mathrm{}$'.format(int(i)), color='k', va='center', ha='right', fontsize=15)
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

    def plot_it(self):
        f = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(self.file['power'],
                        extent=[0, len(self.file['angle']) - 1, np.min(self.file['density']), np.max(self.file['density'])],
                        origin='lower', aspect='auto', cmap='gist_heat')
        self.shade()
        plt.ylabel(r'Electron number density, $n_{\mathrm{e}}$')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        ax1 = plt.subplot(gs[1])
        ax1.plot(self.file['angle'])
        plt.xlim([0, len(self.file['angle']) - 1])
        plt.yticks([30, 45, 60])
        plt.ylabel('Aspect angle')
        axs = []
        axs += [ax0]
        axs += [ax1]
        gs.update(hspace=0.05)
        f.colorbar(im, ax=axs).ax.set_ylabel('Echo power')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        # plt.savefig(f'{self.name}.pgf', bbox_inches='tight', transparent=True)
        plt.show()


if __name__ == '__main__':
    PlotHK().plot_it()
