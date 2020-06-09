"""Main script for  controlling the calculation method of the IS spectrum.
"""

import os
import time
import datetime
import itertools
# The start method of the multiprocessing module was changed from python3.7
# to python3.8. Instead of using 'fork', 'spawn' is the new default.
# To be able to use global variables across all parallel processes,
# the start method must be reset to 'fork'. See
# https://tinyurl.com/yyxxfxst for more info.
import multiprocessing as mp
mp.set_start_method('fork')

import matplotlib  # pylint: disable=C0413
import matplotlib.gridspec as grid_spec  # pylint: disable=C0413
import matplotlib.pyplot as plt  # pylint: disable=C0413
from matplotlib.backends.backend_pdf import PdfPages  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413
import scipy.signal as signal  # pylint: disable=C0413
import si_prefix as sip  # pylint: disable=C0413
import scipy.integrate as si  # pylint: disable=C0413
import scipy.constants as const  # pylint: disable=C0413

from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413
from utils import hello_kitty as hk  # pylint: disable=C0413
from data import reproduce

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class PlotClass:
    """Create a plot object that automatically will show the data created.
    """

    def __init__(self):
        """Make plots of an IS spectrum based on a variety of VDFs.

        Keyword Arguments:
            plasma {bool} -- choose to plot only the part of the
            spectrum where the plasma line is found (default: {False})
        """
        self.save = input('Press "y/yes" to save plot, ' + \
                          'any other key to dismiss.\t').lower()
        self.page = 1
        self.plasma = False
        self.pdffig = None
        self.save_path = str
        self.correct_inputs()
        self.line_styles = ['-', '--', ':', '-.',
                            (0, (3, 5, 1, 5, 1, 5)),
                            (0, (3, 1, 1, 1, 1, 1))]

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        self.correct_inputs()

    def correct_inputs(self):
        """Extra check suppressing the parameters
        that was given but is not necessary.
        """
        try:
            if not isinstance(self.plasma, bool):
                self.plasma = False
        except Exception:
            pass

    def save_it(self, params):
        """Save the figure as a multi page pdf with all
        parameters saved in the meta data.

        The date and time is used in the figure name, in addition
        to it ending with which method was used. The settings that
        was used in config as inputs to the plot object is saved
        in the metadata of the figure.
        """
        version = ''
        for d in params:
            if 'version' in d:
                if any(c.isalpha() for c in version):
                    version += f'_{d["version"][0]}'
                else:
                    version += f'{d["version"][0]}'
        params.insert(0, {'F_MAX': cf.I_P['F_MAX'], 'F0': cf.I_P['F0'],
                          'V_MAX': cf.V_MAX, 'F_N_POINTS': cf.F_N_POINTS,
                          'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS})
        tt = time.localtime()
        the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
        save_path = '../../../report/master-thesis/figures'
        if not os.path.exists(save_path):
            save_path = '../figures'
            os.makedirs(save_path, exist_ok=True)
        self.save_path = f'{save_path}/{the_time}_{version}'
        self.pdffig = PdfPages(str(self.save_path) + '.pdf')
        metadata = self.pdffig.infodict()
        metadata['Title'] = f'ISR Spectrum w/ {version}'
        metadata['Author'] = 'Eirik R. Enger'
        metadata['Subject'] = f"IS spectrum made using a {version} distribution ' + \
                              'and Simpson's integration rule."
        metadata['Keywords'] = f'{params}'
        metadata['ModDate'] = datetime.datetime.today()

    def plot_normal(self, f, Is, func_type, l_txt):
        """Make a plot using f as x axis and Is as y axis.

        Arguments:
            f {np.ndarray} -- variable along x axis
            Is {list} -- list of np.ndarrays that give the y axis
            values along x axis
            func_type {str} -- attribute of the matplotlib.pyplot object
            l_txt {list} -- a list of strings that give the legend
            of the spectra. Same length as the inner lists
        """
        try:
            getattr(plt, func_type)
        except Exception:
            print(f'{func_type} is not an attribute of the ' + \
                  'matplotlib.pyplot object. Using "plot".')
            func_type = 'plot'
        if len(Is) != len(l_txt):
            print('Warning: The number of spectra does ' + \
                  'not match the number of labels.')
        Is = Is.copy()
        # Linear plot show only ion line (kHz range).
        if func_type == 'plot' and not self.plasma:
            f, Is = self.only_ionline(f, Is)
        p, freq, exp = self.scale_f(f)
        plt.figure(figsize=(6, 3))
        if self.plasma:
            # Clip the frequency axis around the plasma frequency.
            mask = self.find_p_line(freq * 10**exp, Is)
            freq = freq[mask]
        if func_type == 'semilogy':
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel('Power [dB]')
            # plt.ylabel(r'$10\times\log_{10}$(Power) [dB]')
            for i, _ in enumerate(Is):
                Is[i] = 10 * np.log10(Is[i])
        else:
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel('Power')
        for st, s, lab in zip(itertools.cycle(self.line_styles), Is, l_txt):
            if self.plasma:
                s = s[mask]
            if func_type == 'semilogy':
                plt.plot(freq, s, 'r', linestyle=st,
                        linewidth=.8, label=lab)
            else:
                plot_object = getattr(plt, func_type)
                plot_object(freq, s, 'r', linestyle=st,
                            linewidth=.8, label=lab)

        plt.legend()
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.tight_layout()

        if self.save in ['y', 'yes']:
            self.pdffig.attach_note(func_type)
            plt.savefig(self.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.save_path) + f'_page_{self.page}.pgf', bbox_inches='tight')
            self.page += 1

    def plot_ridge(self, frequency, multi_parameters, func_type, l_txt, ridge_txt=None):
        """Make a ridge plot of several spectra.

        Arguments:
            frequency {np.ndarray} -- frequency axis
            multi_parameters {list} -- list (outer) containing
            lists (inner) of np.ndarrays.
            The arrays contain the spectrum values at the frequencies
            given by 'frequency'
            func_type {str} -- attribute of the matplotlib.pyplot class
            l_txt {list} -- a list of strings that give the legend of the
            spectra. Same length as the inner lists

        Keyword Arguments:
            ridge_txt {list} -- list of strings that give the text to the left
            of all ridges. Same length as outer list or None (default: {None})
        """
        # Inspired by https://tinyurl.com/y9p5gewr
        try:
            getattr(plt, func_type)
        except Exception:
            print(f'{func_type} is not an attribute of the ' + \
                  'matplotlib.pyplot object. Using "plot".')
            func_type = 'plot'
        if len(multi_parameters) != len(ridge_txt):
            print('Warning: The list of spectra lists is not of the same ' + \
                  'length as the length of "ridge_txt"')
            if len(multi_parameters) > len(ridge_txt):
                for _ in range(len(multi_parameters) - len(ridge_txt)):
                    ridge_txt.append('')
        f_original = frequency.copy()
        multi_params = multi_parameters.copy()
        multi_params.reverse()
        ridge_txt = ridge_txt.copy()
        if ridge_txt is None:
            ridge_txt = ['' for _ in multi_params]
        else:
            ridge_txt.reverse()
        gs = grid_spec.GridSpec(len(multi_params), 1)
        fig = plt.figure(figsize=(7, 9))
        ax_objs = []
        Rgb = np.linspace(0, 1, len(multi_params))
        for j, params in enumerate(multi_params):
            if len(params) != len(l_txt):
                print('Warning: The number of spectra ' + \
                      'does not match the number of labels.')
            # f is reset due to the scaling of 'plot' immediately below.
            f = f_original
            # Linear plot show only ion line (kHz range).
            if func_type == 'plot' and not self.plasma:
                f, params = self.only_ionline(f, params)
            p, freq, exp = self.scale_f(f)
            if self.plasma:
                mask = self.find_p_line(freq * 10**exp, params)
                freq = freq[mask]
            ax_objs.append(fig.add_subplot(gs[j:j + 1, 0:]))
            first = 0
            for st, s, lab in zip(itertools.cycle(self.line_styles), params, l_txt):
                if self.plasma:
                    s = s[mask]
                plot_object = getattr(ax_objs[-1], func_type)
                plot_object(freq, s, color=(Rgb[j], 0., 1 - Rgb[j]), linewidth=1, label=lab, linestyle=st)
                if first == 0:
                    idx = np.argwhere(freq > ax_objs[-1].viewLim.x0)[0]
                    legend_pos = (ax_objs[-1].viewLim.x1, np.max(s))
                    y0 = s[idx]
                    ax_objs[-1].text(freq[idx], s[idx], ridge_txt[j],
                                     fontsize=14, ha="right", va='bottom')
                first += 1
                if j == 0:
                    plt.legend(loc='upper right', bbox_to_anchor=legend_pos, bbox_transform=ax_objs[-1].transData)

            if func_type == 'plot':
                # Make a vertical line of comparable size in all plots.
                self.match_box(f_original, freq, multi_params, [y0, j])

            self.remove_background(ax_objs[-1], multi_params, j, p)

        gs.update(hspace=-0.6)
        if self.save in ['y', 'yes']:
            self.pdffig.attach_note(func_type)
            plt.savefig(self.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.save_path) + f'_page_{self.page}.pgf', bbox_inches='tight')
            self.page += 1

    @staticmethod
    def remove_background(plt_obj, multi_params, j, p):
        # make background transparent
        rect = plt_obj.patch
        rect.set_alpha(0)
        # remove borders, axis ticks and labels
        plt_obj.set_yticklabels([])
        plt.tick_params(axis='y', which='both', left=False,
                        right=False, labelleft=False)
        if j == len(multi_params) - 1:
            plt.xlabel(f'Frequency [{p}Hz]')
        else:
            plt.tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)

        spines = ["top", "right", "left", "bottom"]
        for sp in spines:
            plt_obj.spines[sp].set_visible(False)

    @staticmethod
    def scale_f(frequency):
        """Scale the axis and add the appropriate SI prefix.

        Arguments:
            frequency {np.ndarray} -- the variable along an axis

        Returns:
            str, np.ndarray, int -- the prefix, the scaled variables, the
                                    exponent corresponding to the prefix
        """
        freq = np.copy(frequency)
        exp = sip.split(np.max(freq))[1]
        freq /= 10**exp
        pre = sip.prefix(exp)
        return pre, freq, exp

    @staticmethod
    def find_p_line(freq, spectrum):
        """Find the frequency that is most likely the peak
        of the plasma line and return the lower and upper
        bounds for an interval around the peak.

        Arguments:
            freq {np.ndarray} -- sample points of frequency parameter
            spectrum {list} -- list of np.ndarray, values of spectrum
                               at the sampled frequencies

        Keyword Arguments:
            check {bool} -- used in correct_inputs to check if plasma
                            plots are possible (default: {False})

        Returns:
            np.ndarray -- array with boolean elements
        """
        spec = spectrum[0]
        # Assumes that it is the rightmost peak (highest frequency).
        try:
            p = signal.find_peaks(spec, height=10)[0][-1]
        except Exception:
            print('Warning: did not find any plasma line')
            return freq < np.inf
        f = freq[p]

        lower, upper = f - 1e6, f + 1e6

        # Don't want the ion line to ruin the scaling of the y axis
        if lower < 1e5:
            lower = 1e5
        return (freq > lower) & (freq < upper)

    @staticmethod
    def only_ionline(f, Is):
        Is = Is.copy()
        idx = np.argwhere(abs(f) < 4e4)
        if len(idx) < 3:
            return f, Is
        f = f[idx].reshape((-1,))
        for i, _ in enumerate(Is):
            Is[i] = Is[i][idx].reshape((-1,))
        return f, Is

    def match_box(self, freq_original, freq, multi_parameters, args):
        multi_params = multi_parameters.copy()
        v_line_x = np.linspace(.04, .2, len(multi_params))
        if self.plasma:
            f = freq_original.copy()
            spec = multi_params[0]
            mask = self.find_p_line(f, spec)
        diff = np.inf
        for params in multi_params:
            plot_diff = 0
            for s in params:
                if self.plasma:
                    s = s[mask]
                difference = np.max(s) - np.min(s)
                if plot_diff < difference:
                    plot_diff = difference
            if plot_diff < diff:
                diff = plot_diff

        x0 = np.min(freq) + (np.max(freq) - np.min(freq)) * v_line_x[args[1]]
        plt.vlines(x=x0, ymin=args[0], ymax=args[0] + int(np.ceil(diff / 10) * 5), color='k', linewidth=3)
        plt.text(x0, args[0] + int(np.ceil(diff / 10) * 5) / 2,
                 r'${}$'.format(int(np.ceil(diff / 10) * 5)), rotation=90, ha='right', va='center')


class Simulation:
    def __init__(self):
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
        # self.r = reproduce.PlotTemperature(self.plot)
        self.r = reproduce.PlotHKExtremes(self.plot)

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
        self.r.create_it()
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
                self.plot.save_it(self.meta_data)
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
    # Simulation().run()
    hk.HelloKitty(1).run()
