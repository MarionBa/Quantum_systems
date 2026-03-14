import numpy as np
import Functions as fct
import matplotlib.pyplot as plt

""" Ref. paper: H. El-Ella et al. Vol. 25, No. 13 OPTICS EXPRESS 14809 (2017)"""
""" This script calculates the fluorescence spectrum from the energy states calculated in 'Energy_states.py'. """

# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"


### Fundamental constants ### - Do not modify
omega_0 = 2 * np.pi * 2870 # [MHz]
Ro = 1 # Baseline
p = 10 # Factor for the excitation rate Gamma_p [m^2/J]

### Parameters ### - same parameter as in 'Energy_states.py'
I = 1 * 10 ** 4 # Green laser intensity [W/m^2]

# Import energy states splitting array
parameters = fr'Intensity{I}_Density1000_SpotH1e-07_SpotW1e-07'
Delta = np.load(dir + fr'\Energy splitting arrays\Energy_states_splitting_{parameters}.npy')

# Which nitrogen isotope are your NV centers made of?
Nisotope = 14

if Nisotope == 14:
    # Do not touch
    Stot = 1
    Gamma_p = p * I / 10 ** 6 # [MHz]
    gamma_2 = 2 * np.pi * 0.7 # [MHz]
    # Touch: MW field strength
    Omega = 2 * np.pi * 0.03  # [MHz]

elif Nisotope == 15:
    # Do not touch
    Stot = 1 / 2
    Gamma_p = p * I / 10 ** 6 # [MHz]
    gamma_2 = 2 * np.pi * 1  # [MHz]
    # Touch: MW field strenght
    Omega = 2 * np.pi * 0.1  # [MHz]

else:
    print('Wrong nitrogen isotope!')

# Coupling MW field - array initialization
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)

# Spectra calculation
spectra = []
Delta_extrema = [min(Delta), max(Delta)]
for D in Delta:

    # Shifted energy splitting
    D_GS = 2 * np.pi * D # [MHz]

    spectrum = []
    for omega_c in omega_c_array:
        spectrum.append(fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot))

    spectra.append(spectrum)


#### Figure and save data ###
Y = spectra
x = (omega_c_array-omega_0)/(2*np.pi)
# Upper & lower envelopes
y_upper = np.max(Y, axis=0)
y_lower = np.min(Y, axis=0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, y_upper/max(y_upper), color='crimson', lw=2, label='Upper envelope')
plt.plot(x, y_lower/max(y_lower), color='navy', lw=2, label='Lower envelope')

plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
if Nisotope == 14:
    plt.title(r'Spectrum for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectrum for ${}^{15}N$')

plt.legend()
plt.tight_layout()
plt.savefig(dir + fr'\ODMR_spectrum_N{Nisotope}_{parameters}.png')
plt.show()


# Export final spectrum
np.save(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}.npy', y_lower)


