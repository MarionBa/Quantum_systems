import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample


""" Just a script to plot some of my data: the ODMR contrast, linewidth and the magnetic sensitivity. 
This scripts just loads the above mentioned quantities (calculated in ODMR_spectrum.py scripts and plots the data.
 
 Note: it is not the best way of efficiently plotting data but I just needed a couple of figure so I did not spend
 much time in making this code good looking or time efficient. """

# Directory
dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

# Nitrogen isotope of NV centers
Nisotope = 14

# Import ODMR quatities - import which ever spectra you want to compare
# Contrast
ODMR1_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I4700.0.npy')
ODMR2_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I11614.0.npy')
ODMR3_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I15419.0.npy')
ODMR4_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I36300.0.npy')
ODMR5_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I54971.0.npy')
ODMR6_contrast = np.load(dir + fr'\Energy splitting arrays\ODMR_contrast_N{Nisotope}_I113120.0.npy')
# Linewidth
ODMR1_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I4700.0.npy')
ODMR2_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I11614.0.npy')
ODMR3_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I15419.0.npy')
ODMR4_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I36300.0.npy')
ODMR5_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I54971.0.npy')
ODMR6_linewidth = np.load(dir + fr'\Energy splitting arrays\ODMR_linewidth_N{Nisotope}_I113120.0.npy')
# Magnetic sensitivity
ODMR1 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I4700.0.npy')
ODMR2 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I11614.0.npy')
ODMR3 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I15419.0.npy')
ODMR4 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I36300.0.npy')
ODMR5 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I54971.0.npy')
ODMR6 = np.load(dir + fr'\Energy splitting arrays\ODMR_magnetic_sensitivity_N{Nisotope}_I113120.0.npy')

# Parameter: laser intensity
I = [10 ** 5, 5 * 10 ** 5, 10 ** 6, 5 * 10 ** 6, 10 ** 7, 5 * 10 ** 7]

# Making arrays out of the data (contrast expressed in %)
ODMR_contrast = [ODMR1_contrast*100, ODMR2_contrast*100, ODMR3_contrast*100, ODMR4_contrast*100, ODMR5_contrast*100, ODMR6_contrast*100]
ODMR_linewidth = [np.mean(ODMR1_linewidth), np.mean(ODMR2_linewidth), np.mean(ODMR3_linewidth), np.mean(ODMR4_linewidth), np.mean(ODMR5_linewidth), np.mean(ODMR6_linewidth)]
ODMR = [ODMR1, ODMR2, ODMR3, ODMR4, ODMR5, ODMR6]


### Calculation of laser intensity stability ###
# Here we look at variation of intensity of the green laser and their impact on the ODMR contrast
if Nisotope == 14:
    Fit = max(ODMR_contrast) * np.exp(10e-7 * np.multiply(I, -1)) + min(ODMR_contrast)  # A fit for the ODMR contrast
elif Nisotope == 15:
    # Perform linear fit
    coef = np.polyfit(I, ODMR_contrast, 3)
    coef_1d = [np.float(coef[0]), np.float(coef[1]), np.float(coef[2]), np.float(coef[3])]

    # Create polynomial function
    p = np.poly1d(coef_1d)
    Fit = p(I)

# Ressampling my data
Fit_resamp = resample(Fit, int(10e7))

# Calculation of the standard deviation using a sliding window
n = 0
window = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Std_mean = []
for per in window:
    Std = []
    for i in I:
        w = i * per/100 # Changes of laser intensity [%]
        print(w)

        # Standard deviation calculation
        Std.append(np.std(Fit_resamp[n:n+int(w)]))
        n = n + 1

    # Taking the mean of the standard deviation array, per sliding window
    Std_mean.append(np.mean(Std))

# Print the result
print(Std_mean)


### Figures ###

# Figure of the varying ODMR contrast (standard deviation) vs the green laser intensity
plt.figure()
plt.plot(window, Std_mean, marker='o')
plt.xlabel(r'Changes in green laser intensity [%]')
plt.ylabel('Changes in ODMR contrast [%]')
if Nisotope == 14:
    plt.title(r'Isotope ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Isotope ${}^{15}N$')


# Figure of the ODMR contrast
plt.figure()
plt.semilogx(I, ODMR_contrast, marker='o')
plt.xlabel(r'Intensity [W/$m^2$]')
plt.ylabel('Contrast [%]')
if Nisotope == 14:
    plt.title(r'Contrast for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Contrast for ${}^{15}N$')
#plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e6to310e6.png')

# Figure of the ODMR spectral linewidth
plt.figure()
plt.semilogx(I, ODMR_linewidth, marker='o')
plt.xlabel(r'Intensity [W/$m^2$]')
plt.ylabel('Linewidth [MHz]')
if Nisotope == 14:
    plt.title(r'Linewidth for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Linewidth for ${}^{15}N$')
#plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e6to310e6.png')

# Figure of the ODMR magnetic sensitivity
plt.figure()
plt.semilogx(I, ODMR, marker='o')
#plt.semilogx(I, [ODMR_linewidth[i]/ODMR_contrast[i] for i in range(len(ODMR))], '.')
plt.xlabel(r'Intensity [W/$m^2$]')
plt.ylabel(r'Magnetic sensitivity [T/$\sqrt{Hz}$]')
if Nisotope == 14:
    plt.title(r'Magnetic sensitivity for ${}^{14}N$')
elif Nisotope == 15:
    plt.title(r'Magnetic sensitivity for ${}^{15}N$')
#plt.savefig(dir + fr'\ODMR_spectral_splitting_N{Nisotope}_Intensity10e6to310e6.png')


# Show all figures
plt.show()
