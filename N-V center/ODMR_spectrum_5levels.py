import numpy as np
import Functions as fct
import matplotlib.pyplot as plt
import random
from scipy.signal import peak_widths, find_peaks
from scipy.optimize import curve_fit

""" Ref. paper: H. El-Ella et al. Vol. 25, No. 13 OPTICS EXPRESS 14809 (2017)"""
""" This script calculates the fluorescence spectrum from the energy states calculated in 'Energy_states.py'. """

# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"


### Fundamental constants ### - Do not modify
omega_0 = 2 * np.pi * 2870 # [MHz]
Ro = 7 * 10 ** 7 # fluorescence decay rate: photon counts/s, lowest range for 1 NV center
p = 10 # Factor for the excitation rate Gamma_p [m^2/J]
lat = 0.3567 * 10 ** -9 # [m] diamond lattice spacing
n_eff = 1.6 # Effective refractive index of the mode propagating along the WG
n_medium = 1 # Refractive index of the medium where the evanescent field (does not) propagate
lamb = 675 * 10 ** -9 # Wavelength of the fluorescence field (central wavelength 600-750nm for NV centers
ko = 2 * np.pi / lamb # Wavevector of the fluorescence field
gamma_e = 28 * 10 ** 3 # [MHz] gyromagnetic ratio

# Import energy states splitting array
Delta = np.load(r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results\Energy splitting arrays\Energy_splitting.npy')
Field = np.load(r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results\Energy splitting arrays\Laser_field.npy')
#np.random.shuffle(Field) # create a randomness in the electric field array instead of the evanescent profile which was computed in 'Energy_states.py'

# Which modality
Modality = 'Ev'

if Modality == 'Ev':
    Distance_WG = np.load(
        r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results\Energy splitting arrays\NV_distance.npy')
else:
    print('Confocal microscopy modality, no LDOS considerations.')

# Which nitrogen isotope are your NV centers made of?
Nisotope = 15

# Different parameters depending on your nitrogen isotope
if Nisotope == 14:
    # Do not touch
    Stot = 1
    gamma_2 = 2 * np.pi * 0.7 # [MHz] single NV center 10^4-10^6 s^-1 but take data from ref. paper
    # Touch: MW field strength
    Omega = 2 * np.pi * 0.03  # [MHz]

elif Nisotope == 15:
    # Do not touch
    Stot = 1 / 2
    gamma_2 = 2 * np.pi * 1  # [MHz] single NV center 10^4-10^6 s^-1 but take data from ref. paper
    # Touch: MW field strenght
    Omega = 2 * np.pi * 0.1  # [MHz]

else:
    print('Wrong nitrogen isotope!')

# Coupling MW field - array initialization
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)

# Spectra calculation
spectra = []
FWHM = []
Delta_extrema = [min(Delta), max(Delta)]
n = 0
for D in Delta:

    # Shifted energy splitting
    D_GS = 2 * np.pi * D # [MHz]
    #D_GS = omega_0  # Ground state bare energy N-V center [MHz]

    ### Laser field felt by the NV center: accounts for random orientation of NV center with respect to green laser polarization ###
    # Select random number between 0 and 1 --> gives us the weight of the interaction between laser field and NV center
    Rand_NV_orientation = random.uniform(0, 1)
    Gamma_p = Rand_NV_orientation * p * Field[n] / (10 ** 6) # [MHz]
    print('Decoherence rate:', gamma_2, 'Green laser pump rate:', Gamma_p)

    # Initialization
    spectrum = []
    contrast = []
    # Loop over the MW frequency --> calculates the fluorescence response
    for omega_c in omega_c_array:

        if Modality == 'Ev':

            # Here we account for effective LDOS effect of the distance of the NV center with respect to the WG
            Distance_LDOS = Distance_WG[n] * lat
            Bare_fluorescence = fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot)

            # Effect of evanescent coupling back in the WG
            delta = 1 / (ko * np.sqrt(n_eff ** 2 - n_medium ** 2))  # Evanescent field depth (same as for evanescent field coming from the waveguide, because of reciprocity)
            LDOS_fluorescence = Bare_fluorescence * np.exp(-2*Distance_LDOS/delta)
            spectrum.append(LDOS_fluorescence)
        else:
            spectrum.append(fct.Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot))

    spectra.append(spectrum)
    n = n+1

    ### Calculation of contrast and linewidth ###

    # Contrast
    contrast.append((max(spectrum)-min(spectrum))/max(spectrum))

    # Linewidhth
    x = np.linspace(-len(spectrum)/2, len(spectrum)/2, 1000)
    inv_spectrum = []
    [inv_spectrum.append(-spectrum[i]) for i in range(len(spectrum))]
    norm_spectrum = (inv_spectrum-min(inv_spectrum))/(max(inv_spectrum)-min(inv_spectrum))
    parameters, _ = curve_fit(fct.Gauss, x, norm_spectrum)
    fit_A, fit_B = parameters
    fit_y = fct.Gauss(x, fit_A, fit_B)
    peaks, _ = find_peaks(fit_y)
    half_width = peak_widths(fit_y, peaks, rel_height=0.5)
    FWHM.append(half_width[0])

    ### Test plot ###
    # plt.figure()
    # plt.plot(np.linspace(-len(spectrum)/2, len(spectrum)/2, 1000), norm_spectrum)
    # plt.plot(np.linspace(-len(spectrum)/2, len(spectrum)/2, 1000), fit_y)
    # plt.show()


### Caluclation of magnetic sensitivity ###
eta_B = np.mean(FWHM) / (np.mean(contrast)*gamma_e*np.sqrt(Ro*len(Field)))
print('Magnetic sensitivity:', eta_B)


#### Figure and save data ###

# Export contrast, FWHM,
np.save(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I{np.round(max(Field), 0)}.npy', contrast)
np.save(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I{np.round(max(Field), 0)}.npy', FWHM)
np.save(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I{np.round(max(Field), 0)}.npy', eta_B)


# Figure upper and lower envelopes of all the spectra simulated
Y = spectra
x = (omega_c_array-omega_0)/(2*np.pi)
# Upper & lower envelopes
y_upper = np.max(Y, axis=0)
y_lower = np.min(Y, axis=0)
# Plot
plt.figure(figsize=(8,5))
plt.plot(x, y_upper, color='crimson', lw=2, label='Upper envelope')
plt.plot(x, y_lower, color='navy', lw=2, label='Lower envelope')
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]', fontsize=15)
plt.ylabel('Fluorescence spectrum [photons/sec]', fontsize=15)
if Nisotope == 14:
    plt.title(r'Spectrum for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectrum for ${}^{15}N$')
plt.legend()
plt.xlim(min(x), max(x))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(dir + fr'\ODMR_spectrum_N{Nisotope}.png')


# Figure of all the fluorescence spectra
plt.figure(figsize=(8, 5))
for y in Y:
    plt.plot(x, y)
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]', fontsize=15)
plt.ylabel('Fluorescence spectrum [photons/sec]', fontsize=15)
if Nisotope == 14:
    plt.title(r'NV centers spectra for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'NV centers spectra for ${}^{15}N$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# Figure of the total fluorescence spectrum of all the NV centers
# --> sumation of all the fluorescence traces
plt.figure(figsize=(8, 5))
Spectrum = np.zeros(len(Y[0]))
for y in Y:
    Spectrum = Spectrum + y
plt.plot(x, Spectrum)
if Nisotope == 14:
    plt.title(r'Spectra for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectra for ${}^{15}N$')
plt.xlim(min(x), max(x))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]', fontsize=15)
plt.ylabel('Fluorescence spectrum [photons/sec]', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Export final spectrum
np.save(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_I{max(Field)}.npy', y_lower)


