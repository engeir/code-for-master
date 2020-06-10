"""Main script for  controlling the calculation method of the IS spectrum.
"""

# The start method of the multiprocessing module was changed from python3.7
# to python3.8. Instead of using 'fork', 'spawn' is the new default.
# To be able to use global variables across all parallel processes,
# the start method must be reset to 'fork'. See
# https://tinyurl.com/yyxxfxst for more info.
import multiprocessing as mp
mp.set_start_method('fork')

import matplotlib  # pylint: disable=C0413
import matplotlib.pyplot as plt  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413

from plotting import hello_kitty as hk  # pylint: disable=C0413
from plotting import reproduce  # pylint: disable=C0413
from plotting.plot_class import PlotClass  # pylint: disable=C0413

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class Simulation:
    def __init__(self):
        self.from_file = False
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.plot = PlotClass()
        # self.r = reproduce.PlotTestNumerical(self.plot)
        # self.r = reproduce.PlotTestDebye(self.plot)
        # self.r = reproduce.PlotMaxwell(self.plot)
        # self.r = reproduce.PlotSpectra(self.plot)
        # self.r = reproduce.PlotKappa(self.plot)
        # self.r = reproduce.PlotIonLine(self.plot)
        # self.r = reproduce.PlotPlasmaLine(self.plot)
        self.r = reproduce.PlotTemperature(self.plot)
        # self.r = reproduce.PlotHKExtremes(self.plot)

    def create_data(self):
        """Create IS spectra.

        The spectra should be appended to the self.data list, giving a list
        of spectra that are themselves np.ndarrays, or into a list of such
        lists as the aforementioned.

        A list of spectra can be plotted in 'plot_normal', while a list of
        lists can be plotted by plot_ridge. When using plot_ridge, it is
        assumed that all the lists in the outer list is of equal length.

        The list self.ridge_txt should be the same length as the length
        of the outer list when plotting with plt_ridge, since this text
        will go on the left of every ridge. The list self.legend_txt should
        be the same length as the length of the inner lists, and will give
        the legend for the spectra given in the inner lists.

        Notes:
        Possible items in the sys_set dictionary include:
            B -- Magnetic field strength [T]
            F0 -- Radar frequency [Hz]
            F_MAX -- Range of frequency domain [Hz]
            MI -- Ion mass in atomic mass units [u]
            NE -- Electron number density [m^(-3)]
            NU_E -- Electron collision frequency [Hz]
            NU_I -- Ion collision frequency [Hz]
            T_E -- Electron temperature [K]
            T_I -- Ion temperature [K]
            T_ES -- Temperature of suprathermal electrons in the
                    gauss_shell VDF [K]
            THETA -- Pitch angle [1]
            Z -- Height of real data [100, 599] [km]
            mat_file -- Important when using real data and decides
                        the time of day
            pitch_angle -- list of integers that determine which slices
                           of the pitch angles are used. 'all' uses all

        Examples:
        ::
            TEMPS = [2000, 5000]
            methods = ['maxwell', 'kappa']
            sys_set = {'B': 5e-4, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0,
                       'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                       'THETA': 40 * np.pi / 180, 'Z': 599,
                        'mat_file': 'fe_zmuE-01.mat'}
            params = {'kappa': 3, 'vdf': 'kappa', 'area': False}
            for T in TEMPS:
                ridge = []
                sys_set['T_E'] = T
                self.ridge_txt.append(f'$T_e = {T}$ K')
                for m in methods:
                    self.f, s, meta_data = isr.isr_spectrum(m, sys_set, **params)
                    self.meta_data.append(meta_data)
                    ridge.append(s)
                self.data.append(ridge)

            # For a nicer legend, it is added manually
            self.legend_txt.append('Maxwellian')
            self.legend_txt.append('Kappa')
        """
        self.from_file = True
        self.r.create_it('../figures/', 'temp_ridge.npz', from_file=self.from_file)
        self.f = self.r.f
        self.data = self.r.data
        self.legend_txt = self.r.legend_txt
        self.ridge_txt = self.r.ridge_txt
        self.meta_data = self.r.meta_data

    def plot_data(self):
        """Plot the created data from self.data.

        If you want to only plot the plasma line, set
        self.plot.plasma = True

        self.plot.plot_normal() accepts list of np.ndarray and
        self.plot.plot_ridge() accepts list of lists of np.ndarray,
        i.e. list of the structure you send to self.plot.plot_normal()

        Examples:
        ::
            # Given the example in self.create_data()
            # self.plot.plasma = True
            self.plot.plot_normal(self.f, self.data[0], 'plot',
                                  self.legend_txt)
            self.plot.plot_normal(self.f, self.data[0], 'semilogy',
                                  self.legend_txt)
            self.plot.plot_ridge(self.f, self.data, 'plot', self.legend_txt,
                                 self.ridge_txt)
            self.plot.plot_ridge(self.f, self.data, 'semilogy',
                                 self.legend_txt, self.ridge_txt)
        """
        self.r.plot_it()

    def save_handle(self, mode):
        if mode == 'setUp':
            if self.plot.save in ['y', 'yes']:
                self.plot.save_it(self.f, self.data, self.legend_txt, self.ridge_txt, self.meta_data)
        elif mode == 'tearDown':
            if self.plot.save in ['y', 'yes']:
                self.plot.pdffig.close()
            plt.show()

    def run(self):
        self.create_data()
        self.save_handle('setUp')
        self.plot_data()
        self.save_handle('tearDown')


if __name__ == '__main__':
    Simulation().run()
    # hk.HelloKitty(1).run()
