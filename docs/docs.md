<h1 align="center">ISR spectrum</h1>

---

## Contents
- [Info](#info)
- [Installing](#installing)
- [Usage](#usage)
- [File structure](#structure)

## Info <a name = "info"></a>
Calculates an incoherent scatter radar spectrum based on the theory presented in [Hagfors (1961)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699) and [Mace (2003)](https://aip.scitation.org/doi/pdf/10.1063/1.1570828).

## Installing <a name = "installing"></a>
The program is built using `python3.8` and tested on **macOS**.

Clone the repository using `git clone https://github.com/engeir/isr_spectrum.git` or download the latest release, [v1.0](https://github.com/engeir/isr_spectrum/archive/v1.0.zip).
Run
```
pip install -r requirements.txt
```
from directory `isr_spectrum` where `requirements.txt` is located to install all needed packages, and then navigate to the `program` folder:
```
cd program
```
Start the program with command
```
python3 main.py
```

## Usage <a name = "usage"></a>
### Physical environment
The available plasma parameters that can be changed within the program are
```
=== Input parameters ===
K_RADAR -- Radar wavenumber (= -4pi(radar frequency)/(speed of light)) [m^(-1)]
B -- Magnetic field strength [T]
MI -- Ion mass in atomic mass units [u]
NE -- Electron number density [m^(-3)]
NU_E -- Electron collision frequency [Hz]
NU_I -- Ion collision frequency [Hz]
T_E -- Electron temperature [K]
T_I -- Ion temperature [K]
T_ES -- Temperature of suprathermal electrons in the gauss_shell VDF [K] (no longer in use)
THETA -- Aspect angle [1]
Z -- Height used for calculated distribution [100, 599] [km]
mat_file -- Time of day for calculated distribution
pitch_angle -- Pitch angle for calculated distribution
```
which are given in `main.py`.

### Calculation method
The program support different methods of calculating the spectrum, based on how you assume the particles to be distributed. This includes a Maxwellian distribution and a kappa distribution, in addition to any arbitrary isotropic distribution.

The version that determine the calculation method is described in `spectrum_calculation.py`, in the docstring of the function `isr_spectrum`, with additional keyword arguments.

## File structure <a name = "structure"></a>
```
isr_spectrum/
├── program/
│   ├── data/
│   │   └── arecibo/
│   │       ├── E4fe.dat
│   │       ├── fe_zmuE-07.mat
│   │       ├── SzeN.dat
│   │       ├── theta_lims.dat
│   │       ├── timeOfDayUT.dat
│   │       └── z4fe.dat
│   ├── inputs/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── main.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── hello_kitty.py
│   │   ├── plot_class.py
│   │   └── reproduce.py
│   ├── test/
│   │   ├── __init__.py
│   │   └── test_ISR.py
│   └── utils/
│       ├── __init__.py
│       ├── parallel/
│       │   ├── __init__.py
│       │   ├── gordeyev_int_parallel.py
│       │   └── v_int_parallel.py
│       ├── integrand_functions.py
│       ├── read.py
│       ├── spectrum_calculation.py
│       └── vdfs.py
├── LICENSE
├── readme.md
└── requirements.txt
```
