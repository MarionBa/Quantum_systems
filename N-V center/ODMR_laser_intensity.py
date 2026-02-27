import numpy as np
import Functions as fct
import matplotlib.pyplot as plt


# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

# Nitrogen isotope of NV centers
Nisotope = 14

# Import ODMR spectra - import which ever spectra you want to compare

# Intensity laser 13 * 10^8 W/m^2
parameters = 'Intensity1300000000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR1 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}.npy')

# Intensity laser 13 * 10^6 W/m^2
parameters = 'Intensity13000000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR2 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}.npy')

# Intensity laser 13 * 10^4 W/m^2
parameters = 'Intensity130000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR3 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}.npy')

### Parameters ###

omega_0 = 2 * np.pi * 2870 # [MHz]
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)


# Figure
plt.figure()
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR1/max(ODMR1))
# plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR2/max(ODMR2))
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR3/max(ODMR3))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
if Nisotope == 14:
    plt.title(r'Spectra variation for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectra variation for ${}^{15}N$')
plt.xlim(-10, 10)
plt.legend([r'$I=1.3kW/m^2$', r'$I=13MW/m^2$'])
plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e4to10e8.png')
plt.show()
