import numpy as np
import Functions as fct
import matplotlib.pyplot as plt


# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

# Nitrogen isotope of NV centers
Nisotope = 15

# Import ODMR spectra - import which ever spectra you want to compare

# Intensity laser 10 * 10^4 W/m^2
parameters = 'Intensity1000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR1 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}_ModalityConfMic.npy')

# Intensity laser 20 * 10^4 W/m^2
parameters = 'Intensity2000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR2 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}_ModalityConfMic.npy')

# Intensity laser 50 * 10^4 W/m^2
parameters = 'Intensity3000_Density1000_SpotH1e-07_SpotW1e-07'
ODMR3 = np.load(dir + fr'\Energy splitting arrays\ODMR_spectrum_N{Nisotope}_{parameters}_ModalityConfMic.npy')



### Parameters ###

omega_0 = 2 * np.pi * 2870 # [MHz]
omega_c_array = np.linspace(omega_0-100, omega_0+100, 1000)


# Figure
plt.figure()
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR1/max(ODMR1))
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR2/max(ODMR2))
plt.plot((omega_c_array-omega_0)/(2*np.pi), ODMR3/max(ODMR3))
plt.xlabel(r'Detuning $(\omega_c-\omega_0)/2\pi$ [MHz]')
plt.ylabel('Fluorescence spectrum [a.u.]')
if Nisotope == 14:
    plt.title(r'Spectra variation for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Spectra variation for ${}^{15}N$')
plt.xlim(-10, 10)
plt.legend([r'$I=10kW/m^2$', r'$I=20kW/m^2$', r'$I=50kW/m^2$'])
plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e4to10e6.png')
plt.show()
