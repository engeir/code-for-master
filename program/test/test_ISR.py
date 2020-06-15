"""This script implements tests for
functions used throughout the program.

Run from directory `program` with command
python -m unittest test.test_ISR -b
"""

import multiprocessing as mp
mp.set_start_method('fork')

import unittest  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413
import scipy.integrate as si  # pylint: disable=C0413
import scipy.constants as const  # pylint: disable=C0413

from utils import spectrum_calculation as isr  # pylint: disable=C0413
from utils import vdfs  # pylint: disable=C0413


class TestISR(unittest.TestCase):
    """Check if the output from isr_spectrum is as expected.

    Should return two numpy.ndarrays of equal shape.

    Arguments:
        unittest.TestCase {class} -- inherits from unittest
        to make it a TestCase
    """

    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = None, None

    def setUp(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c
        self.sys_set = {'K_RADAR': K_RADAR, 'B': 5e-4, 'MI': 16,
                        'NE': 2e11, 'NU_E': 0, 'NU_I': 0,
                        'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                        'THETA': 40 * np.pi / 180, 'Z': 599,
                        'mat_file': 'fe_zmuE-07.mat', 'pitch_angle': 'all'}
        self.params = {'kappa': 3, 'vdf': 'gauss_shell', 'area': False}

    def tearDown(self):
        self.assertIsInstance(self.a, np.ndarray)
        self.assertIsInstance(self.b, np.ndarray)
        self.assertEqual(self.a.shape, self.b.shape, msg='a.shape != b.shape')

    def test_isr_maxwell(self):
        self.a, self.b, meta_data = isr.isr_spectrum('maxwell', self.sys_set, **self.params)
        self.assertEqual(meta_data['kappa'], None)
        self.assertEqual(meta_data['vdf'], None)
        self.assertEqual(meta_data['T_ES'], None)
        self.assertEqual(meta_data['Z'], None)
        self.assertEqual(meta_data['mat_file'], None)

    def test_isr_kappa(self):
        self.a, self.b, meta_data = isr.isr_spectrum('kappa', self.sys_set, **self.params)
        self.assertEqual(meta_data['kappa'], 3)
        self.assertEqual(meta_data['vdf'], None)
        self.assertEqual(meta_data['T_ES'], None)
        self.assertEqual(meta_data['Z'], None)
        self.assertEqual(meta_data['mat_file'], None)

    def test_isr_long_calc_gauss(self):
        self.a, self.b, meta_data = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)
        self.assertEqual(meta_data['kappa'], None)
        self.assertEqual(meta_data['vdf'], 'gauss_shell')
        self.assertEqual(meta_data['T_ES'], 90000)
        self.assertEqual(meta_data['Z'], None)
        self.assertEqual(meta_data['mat_file'], None)

    def test_isr_long_calc_real(self):
        self.params['vdf'] = 'real_data'
        self.a, self.b, meta_data = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)
        self.assertEqual(meta_data['kappa'], None)
        self.assertEqual(meta_data['vdf'], 'real_data')
        self.assertEqual(meta_data['T_ES'], None)
        self.assertEqual(meta_data['Z'], 599)
        self.assertEqual(meta_data['mat_file'], 'fe_zmuE-07.mat')


# Reference to TestVDF $\label{lst:testVDF}$
class TestVDF(unittest.TestCase):
    """Class which test if the VDFs are normalized.

    Arguments:
        unittest.TestCase {class} -- inherits from unittest
        to make it a TestCase
    """

    @classmethod
    def setUpClass(cls):
        cls.v = np.linspace(0, (6e6)**(1 / 3), int(4e4))**3
        cls.params = {'m': 9.1093837015e-31, 'T': 1000,
                      'kappa': 3, 'T_ES': 90000, 'Z': 300,
                      'mat_file': 'fe_zmuE-07.mat', 'pitch_angle': 'all'}
        cls.f = None
        # cls.fs = []

    # @classmethod
    # def tearDownClass(cls):
    #     np.savez('f', v=cls.v, m=cls.fs[0], k=cls.fs[1], r=cls.fs[2])

    def tearDown(self):
        # The function f is scaled with the Jacobian of cartesian to spherical
        f = self.f.f_0() * self.v**2 * 4 * np.pi
        res = si.simps(f, self.v)
        self.assertAlmostEqual(res, 1, places=6)

    def test_vdf_maxwell(self):
        self.f = vdfs.F_MAXWELL(self.v, self.params)
        # self.fs.insert(0, self.f.f_0())

    def test_vdf_kappa(self):
        self.f = vdfs.F_KAPPA(self.v, self.params)
        # self.fs.insert(1, self.f.f_0())

    # def test_vdf_kappa_vol2(self):
    #     self.f = vdfs.F_KAPPA_2(self.v, self.params)

    # def test_vdf_gauss_shell(self):
    #     self.f = vdfs.F_GAUSS_SHELL(self.v, self.params)

    def test_vdf_real_data(self):
        self.f = vdfs.F_REAL_DATA(self.v, self.params)
        # self.fs.insert(2, self.f.f_0())


if __name__ == '__main__':
    unittest.main()
