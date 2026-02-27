import numpy as np
import Functions as fct
import matplotlib.pyplot as plt


# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

# Import energy states splitting array
parameters = 'Intensity13000000_Density1000_SpotH1e-07_SpotW1e-07'
Delta = np.load(dir + fr'\Energy splitting arrays\Energy_states_splitting_{parameters}.npy')

# Which nitrogen isotope are your NV centers made of?
Nisotope = 15


### Parameters ###

omega_0 = 2 * np.pi * 2870 # [MHz]
Ro = 1 # Baseline

if Nisotope == 14:
    # Do not touch
    Stot = 1
    Gamma_p = 2 * np.pi * 0.02 # [MHz]
    gamma_2 = 2 * np.pi * 0.7 # [MHz]
    # Touch: MW field strength
    Omega = 2 * np.pi * 0.03  # [MHz]

elif Nisotope == 15:
    # Do not touch
    Stot = 1 / 2
    Gamma_p = 2 * np.pi * 0.05  # [MHz]
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


