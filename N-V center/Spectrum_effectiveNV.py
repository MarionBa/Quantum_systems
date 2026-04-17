import numpy as np
import Functions as fct
import matplotlib.pyplot as plt

""" Script which reproduces the ODMR spectrum as shown in the reference paper:
H. El-Ella et al. “Optimised frequency modulation for continuous-wave optical
 magnetic resonance sensing using nitrogen-vacancy ensembles” Vol. 25, No. 13 OPTICS EXPRESS (2017)
 using the parameters of the paper """

### 14N ###
Stot = 1
Gamma_p = 2 * np.pi * 0.02 # [MHz]
Omega = 2 * np.pi * 0.03 # [MHz]
gamma_2 = 2 * np.pi * 0.7 # [MHz]
omega_0 = 2 * np.pi * 2870 # [MHz]
Ro = 1 # Baseline
D_GS = 2 * np.pi * 2.87 * 10 ** 3 # [GHz]

# Coupling field
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)

# Spectrum calculation
spectrum = []
for omega_c in omega_c_array:
    spectrum.append(fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot))


# Figure 1
plt.figure()
plt.plot((omega_c_array-omega_0)/(2*np.pi), spectrum/max(spectrum))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
plt.title(r'Spectrum for ${}^{14}N$')
plt.show()


### 15N ###
Stot = 1/2
Gamma_p = 2 * np.pi * 0.05 # [MHz]
Omega = 2 * np.pi * 0.1 # [MHz]
gamma_2 = 2 * np.pi * 1 # [MHz]
omega_0 = 2 * np.pi * 2870 # [MHz]

# Coupling field
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)

# Spectrum calculation
spectrum = []
for omega_c in omega_c_array:
    spectrum.append(fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot))


# Figure 1
plt.figure()
plt.plot((omega_c_array-omega_0)/(2*np.pi), spectrum/max(spectrum))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
plt.title(r'Spectrum for ${}^{15}N$')
plt.show()