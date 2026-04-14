import numpy as np
import Functions as fct
import matplotlib.pyplot as plt


# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

# Nitrogen isotope of NV centers
Nisotope = 15

# Import ODMR spectra - import which ever spectra you want to compare

# Intensity laser 1 * 10^6 W/m^2
ODMR1 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_I17679.554599410305.npy')

# Intensity laser 3 * 10^6 W/m^2
ODMR2 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_I30621.88682136668.npy')


### Parameters ###

omega_0 = 2 * np.pi * 2870 # [MHz]
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)


# Figure
plt.figure()
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR1/max(ODMR1))
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR2/max(ODMR2))
# plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR3/max(ODMR3))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
if Nisotope == 14:
    plt.title(r'Spectra variation for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectra variation for ${}^{15}N$')
plt.xlim(-10, 10)
plt.legend([r'$I=1MW/m^2$', r'$I=3MW/m^2$']) #, r'$I=50kW/m^2$'])
plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e6to310e6.png')
plt.show()
