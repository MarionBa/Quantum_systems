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

# Import energy states splitting array
Delta = np.load(r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results\Energy splitting arrays\Energy_splitting.npy')
Field = np.load(r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results\Energy splitting arrays\Laser_field.npy')

# Which nitrogen isotope are your NV centers made of?
Nisotope = 15

if Nisotope == 14:
    # Do not touch
    Stot = 1
    gamma_2 = 2 * np.pi * 0.7 # [MHz]
    # Touch: MW field strength
    Omega = 2 * np.pi * 0.03  # [MHz]


elif Nisotope == 15:
    # Do not touch
    Stot = 1 / 2
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
n = 0
for D in Delta:

    # Shifted energy splitting
    D_GS = 2 * np.pi * D # [MHz]
    #D_GS = omega_0  # Ground state bare energy N-V center [MHz]

    # Laser field felt by the NV center
    Gamma_p = p * Field[n] / 10 ** 6 # [MHz]

    spectrum = []
    for omega_c in omega_c_array:
        spectrum.append(fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot))

    spectra.append(spectrum)
    n = n+1


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
#plt.plot(x, y_lower/max(y_lower), color='maroon', lw=2)
print(y_upper-y_lower)

plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]', fontsize=15)
plt.ylabel('Fluorescence spectrum [a.u.]', fontsize=15)
if Nisotope == 14:
    plt.title(r'Spectrum for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectrum for ${}^{15}N$')

#plt.legend()
plt.xlim(min(x), max(x))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(dir + fr'\ODMR_spectrum_N{Nisotope}.png')
plt.show()


# Export final spectrum
np.save(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}.npy', y_lower)


