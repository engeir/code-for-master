import os
import time
import datetime
import itertools

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.signal as signal
import si_prefix as sip

from inputs import config as cf

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
        # np.savez(f'{save_path}', frequency=f, spectra=data, legend_txt=l_txt, ridge_txt=r_txt, meta=params)
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
