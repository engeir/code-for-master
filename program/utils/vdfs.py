"""Velocity distribution function used in the version a_vdf, one of the integrands available for use in the Gordeyev integral.

Any new VDF must be added as an option in the a_vdf function in integrand_functions.py.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.constants as const
import scipy.special as sps
import scipy.integrate as si

from inputs import config as cf
from data import read


class VDF(ABC):
    """All VDF classes must have a method that returns a distribution function.

    Arguments:
        ABC {class} -- make it an abstract base class that all VDF objects should inherit from
    """
    @abstractmethod
    def normalize(self):
        """Calculate the normalization for the VDF.
        """

    @abstractmethod
    def f_0(self):
        """Return the values along the velocity axis of a VDF.
        """


class F_MAXWELL(VDF):
    """Create an object to make Maxwellian distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make sure crucial methods are included
    """
    def __init__(self, v, params):
        self.v = v
        self.params = params
        self.normalize()

    def normalize(self):
        self.A = (2 * np.pi * self.params['T'] * const.k / self.params['m'])**(- 3 / 2)

    def f_0(self):
        func = self.A * np.exp(- self.v**2 / (2 * self.params['T'] * const.k / self.params['m']))

        return func


class F_KAPPA(VDF):
    """Create an object to make kappa distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make sure crucial methods are included
    """
    def __init__(self, v, params):
        """Initialize VDF parameters.

        Arguments:
            v {np.ndarray} -- 1D array with the sampled velocities
            params {dict} -- a dictionary with all needed plasma parameters
        """
        self.v = v
        self.params = params
        self.normalize()

    def normalize(self):
        self.theta_2 = 2 * ((self.params['kappa'] - 3 / 2) / self.params['kappa']) * self.params['T'] * const.k / self.params['m']
        self.A = (np.pi * self.params['kappa'] * self.theta_2)**(- 3 / 2) * \
            sps.gamma(self.params['kappa'] + 1) / sps.gamma(self.params['kappa'] - 1 / 2)

    def f_0(self):
        """Return the values along velocity v of a kappa VDF.

        Kappa VDF used in Gordeyev paper by Mace (2003).

        Returns:
            np.ndarray -- 1D array with the VDF values at the sampled points
        """
        func = self.A * (1 + self.v**2 / (self.params['kappa'] * self.theta_2))**(- self.params['kappa'] - 1)

        return func


class F_KAPPA_2(VDF):
    """Create an object to make kappa vol. 2 distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class for VDF object
    """
    def __init__(self, v, params):
        """Initialize VDF parameters.

        Arguments:
            v {np.ndarray} -- 1D array with the sampled velocities
            params {dict} -- a dictionary with all needed plasma parameters
        """
        self.v = v
        self.params = params
        self.normalize()

    def normalize(self):
        self.v_th = np.sqrt(self.params['T'] * const.k / self.params['m'])
        self.A = (np.pi * self.params['kappa'] * self.v_th**2)**(- 3 / 2) * \
            sps.gamma(self.params['kappa']) / sps.gamma(self.params['kappa'] - 3 / 2)

    def f_0(self):
        """Return the values along velocity v of a kappa VDF.

        Kappa VDF used in dispersion relation paper by Ziebell, Gaelzer and Simoes (2017).
        Defined by Leubner (2002) (sec 3.2).

        Returns:
            np.ndarray -- 1D array with the VDF values at the sampled points
        """
        func = self.A * (1 + self.v**2 / (self.params['kappa'] * self.v_th**2))**(- self.params['kappa'])

        return func


class F_GAUSS_SHELL(VDF):
    def __init__(self, v, params):
        self.v = v
        self.params = params
        self.vth = np.sqrt(self.params['T'] * const.k / self.params['m'])
        self.r = (self.params['T_ES'] * const.k / self.params['m'])**.5
        self.f_M = F_MAXWELL(self.v, self.params)
        self.normalize()

    def normalize(self):
        func = np.exp(- 2 * (abs(self.v) - self.r)**2 / (2 * self.params['T'] * const.k / self.params['m']))
        f = func * self.v**2 * 4 * np.pi
        self.A = 1 / si.simps(f, self.v)
        print(f'Gauss shell at v = {round(self.r / self.vth, 3)} v_th')

    def f_0(self):
        func = self.A * np.exp(- 2 * (abs(self.v) - self.r)**2 / (2 * self.params['T'] * const.k / self.params['m'])) + 10 * self.f_M.f_0()

        return func / 11


class F_REAL_DATA(VDF):
    def __init__(self, v, params):
        self.v = v
        self.params = params
        self.normalize()

    def normalize(self):
        func = read.interpolate_data(self.v, self.params)
        f = func * self.v**2 * 4 * np.pi
        self.A = 1 / si.simps(f, self.v)

    def f_0(self):
        func = self.A * read.interpolate_data(self.v, self.params)

        return func
