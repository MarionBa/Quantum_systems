import numpy as np
import Functions as fct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


""" Optimization of Rydberg EIT peak height - Section 4.3 of the reference paper 
H-J Su et al. "Optimizing the Rydberg EIT spectrum in a thermal vapor" Optics Express 30,2 (2022) 
Note: there are no indication in the ref. paper about the optical density (OD) \alpha and the temperature scaling
nore is there information about the distribution of temperatures (OD) over the temperature (OD) range in Fig. 4.
Thus, I make my best extrapolation based on fitting the model to their results in Fig. 4.
This code has variations to what is described in the reference paper:
1- The Rydberg relaxation parameter 2*\pi*Gamma_r is set at 0.05MHz instead of 5MHz
2- A booster of the peaks height is applied: Boost_peak = 1.5 * 10 ** 4 
Note: I had to guess the temperature steps, the corresponding alpha parameters as well as the probe intensity
which is not well indicated in the ref. paper.
"""

### Saving directory, for figures ###
dir = r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\Rydberg atom\Results EIT peak optimization'

### General parameters of the system ###

# Parameters in ref. paper for Fig. 3
Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]

Gamma_e = 2 * np.pi * 60 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 0.08 # Spontaneous emission Rydberg state [MHz]
Omega_c_CGC1 = 2 * np.pi * 0.38 # The Rabi frequency of the coupling laser for CGC = 1 [MHz]
gamma = 2 * np.pi * 0.75 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe [MHz]
phi = 0
# Magnetic parameters as in ref. paper
gF_g = 1/2 # Lande factor ground state
gF_e = 2/3 # Lande factor excited state
mu_B_over_hbar = 1 # [MHz/G]
B = 0


# Optical density variation --> scales as \alpha \propto exp(-1/T), alpha values extrapolates for best fit with
# results of ref. paper Fig. 4
Alpha = [0.42, 0.7, 1.2, 1.8, 3, 5]

# Intensity probe field = I/I_sat, see ref.
# I tried to guess the field intensity with respect of the units used in [1]
I = 0.3 * np.geomspace(5 * 10 ** -4, 15, num=100)

# Transition |e> state
Gamma_e_array = np.linspace(3.9*Gamma_e, 1.8*Gamma_e, 6)

# Initialization arrays
T_array = []
Peak_array = []
Baseline_array = []

# Loop over the optical density
n = 0
for alpha in Alpha:
    n = n+1

    print(fr'Computing {(n-1)*100/(len(Alpha)-1)}% done')
    print(fr'Optical density: {alpha}')

    ################################################################

    # Initialization arrays
    T_I = []
    Peak_I = []
    Baseline_I = []

    # Loop over the probe field intensity
    for I_idx in I:

        # Probe field intensity
        ap = 1 # If population in state |1>, \sqrt[7/15] if population distributed in all Zeeman |g> states
        Gamma_renorm = Gamma_e_array[n-1] # Renormalized \Gamma_e with respect to temperature
        Omega_p = ap * Gamma_renorm * np.sqrt(1/2) * np.sqrt(I_idx)   # The Rabi frequency of the probe laser [MHz] see ref. paper


        #### Our 5/2 EIT peaks ####

        # Parameters for transitions to |r> states |3> and |8>
        CG_3 = 1 * 2/3 # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
        Emp_fact = 1.5 # Empirical factor for fitting experimental data, from ref. paper
        Omega_c = CG_3 * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
        Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2>
        gF_r = 3 * 1.2 # Ad hoc Lande factor to control Zeeman splitting of the largest EIT peak

        ### |1> --> |2> --> |3> transition ###

        # Magnetic parameters
        mF_g = 2
        Delta_g = mu_B_over_hbar * gF_g * mF_g
        mF_e = 3
        Delta_e = mu_B_over_hbar * gF_e * mF_e
        mF_r = 4
        Delta_r = mu_B_over_hbar * gF_r * mF_r

        # Absorption cross-section
        sigma_23 = fct.Sigma(Delta_c, Delta_p, Delta_s, Delta_g, Delta_e, Delta_r, Omega_c, Omega_p, Gamma_e, Gamma_r, gamma, phi, B)

        #######################################################################

        # Parameters for transitions to |r> states |4> and |9>
        CG_4 = 0.32 * 2/3  # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
        Omega_c = CG_4 * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
        gF_r = 0.1 # Ad hoc Lande factor to avoid large Zeeman splitting and to fit ref. data better

        ### The |1> --> |2> --> |4> transition ###

        # Magnetic components
        mF_g = 2
        Delta_g = mu_B_over_hbar * gF_g * mF_g
        mF_e = 3
        Delta_e = mu_B_over_hbar * gF_e * mF_e
        mF_r = 2
        Delta_r = mu_B_over_hbar * gF_r * mF_r

        # Absorption cross-section
        sigma_24 = fct.Sigma(Delta_c, Delta_p, Delta_s, Delta_g, Delta_e, Delta_r, Omega_c, Omega_p, Gamma_e, Gamma_r, gamma, phi, B)

        ################################################################
        ################################################################

        #### Our 3/2 EIT peak ####

        # Parameters for transitions to |r> states |5> and |10>
        CG_5 = 0.78 * 1/3 # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
        Omega_c = CG_5 * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz] (without empirical booster, as in ref.)
        Delta_s = 0 # EIT transition |33D_3/2>
        gF_r = 0.8

        ### The |1> --> |2> --> |5> transition ###

        # Magnetic components
        Delta_g = 0
        Delta_e = 0
        Delta_r = 0

        # Absorption cross-section
        sigma_25 = fct.Sigma(Delta_c, Delta_p, Delta_s, Delta_g, Delta_e, Delta_r, Omega_c, Omega_p, Gamma_e, Gamma_r, gamma, phi, B)

        #######################################################################
        #######################################################################

        ### Total transmission - \sgima_eg = product of sigma_ij ###
        T = np.exp(-alpha*(np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)))
        T_I.append(T)

        #######################################################################
        #######################################################################

        ### Extracting EIT peak heights and width ###
        Boost_peak = 1.5 * 10 ** 4
        peaks, _ = find_peaks(T, height=0)
        if len(peaks) == 1:
            Peak_I.append((T[peaks[0]]-min(T))*Boost_peak)
        elif len(peaks) == 0:
            print('Ho no, you have no peaks detected!')
        else:
            Peak_I.append((T[peaks[1]] - min(T))*Boost_peak)

        #######################################################################
        #######################################################################

        ### Extracting baseline transmission ###

        Omega_p = ap * Gamma_renorm * np.sqrt(1 / 2) * np.sqrt(I_idx) # Probe frequency without booster for baseline
        Baseline_I.append(np.exp(-alpha*np.square(Gamma_e)/(np.square(Gamma_e)+2*np.square(Omega_p)))*100)
        # Baseline transmission in %


    # Save transmission spectra in tensor
    T_array.append(T_I)
    Peak_array.append(Peak_I)
    Baseline_array.append(Baseline_I)

### Figures ###

# Renormalization laser field intensity for correct laser fluence in Fig 4 (b) of ref. paper
I = I * 0.2

# Temperature corresponding to the ODs (alpha)
Temp = [27, 35, 43, 51, 58, 65]

# Figure 1: transmission baseline
plt.figure(figsize=(8, 6))
plt.semilogx(I, Baseline_array[0], color='tab:blue', linewidth=3)
plt.semilogx(I, Baseline_array[1], color='tab:green', linewidth=3)
plt.semilogx(I, Baseline_array[2], color='tab:red', linewidth=3)
plt.semilogx(I, Baseline_array[3], color='tab:brown', linewidth=3)
plt.semilogx(I, Baseline_array[4], color='k', linewidth=3)
plt.semilogx(I, Baseline_array[5], color='tab:pink', linewidth=3)
#plt.legend([fr'$\alpha =$ {Alpha[0]}', fr'$\alpha =$ {Alpha[1]}', fr'$\alpha =$ {Alpha[2]}', fr'$\alpha =$ {Alpha[3]}', fr'$\alpha =$ {Alpha[4]}', fr'$\alpha =$ {Alpha[5]}'])
plt.legend([r'T = ' + str(Temp[0]) + ' $^{\circ}$C', fr'T = ' + str(Temp[1]) + ' $^{\circ}$C', fr'T = ' + str(Temp[2]) + ' $^{\circ}$C', fr'T = ' + str(Temp[3]) + ' $^{\circ}$C', fr'T = ' + str(Temp[4]) + ' $^{\circ}$C', fr'T = ' + str(Temp[5]) + ' $^{\circ}$C'])
plt.xlabel(r'Probe intensity [W/cm$^2$]', fontsize = 15)
plt.ylabel(r'$T_B$ [%]', fontsize = 15)
plt.xlim([min(I), max(I)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(dir + r'\Temperature and probe intensity\Baseline.png')

# Figure 2: big EIT peak height for different temperatures as function of probe intensity
plt.figure(figsize=(8, 6))
plt.semilogx(I, Peak_array[0], color='tab:blue', linewidth=3)
plt.semilogx(I, Peak_array[1], color='tab:green', linewidth=3)
plt.semilogx(I, Peak_array[2], color='tab:red', linewidth=3)
plt.semilogx(I, Peak_array[3], color='tab:brown', linewidth=3)
plt.semilogx(I, Peak_array[4], color='k', linewidth=3)
plt.semilogx(I, Peak_array[5], color='tab:pink', linewidth=3)
#plt.legend([fr'$\alpha =$ {Alpha[0]}', fr'$\alpha =$ {Alpha[1]}', fr'$\alpha =$ {Alpha[2]}', fr'$\alpha =$ {Alpha[3]}', fr'$\alpha =$ {Alpha[4]}', fr'$\alpha =$ {Alpha[5]}'])
plt.legend([r'T = ' + str(Temp[0]) + ' $^{\circ}$C', fr'T = ' + str(Temp[1]) + ' $^{\circ}$C', fr'T = ' + str(Temp[2]) + ' $^{\circ}$C', fr'T = ' + str(Temp[3]) + ' $^{\circ}$C', fr'T = ' + str(Temp[4]) + ' $^{\circ}$C', fr'T = ' + str(Temp[5]) + ' $^{\circ}$C'])
plt.xlabel(r'Probe intensity [W/cm$^2$]', fontsize = 15)
plt.ylabel(r'Rydberg EIT peak height [%]', fontsize = 15)
plt.xlim([min(I), max(I)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(dir + r'\Temperature and probe intensity\EIT_peaks.png')

# Figure 3: alpha plot vs T
# plt.figure()
# #plt.plot(T, Alpha)
# plt.plot(T, 7 * np.exp(-50/np.array(T)))
# plt.ylabel(r'$\alpha$')
# plt.xlabel(r'T [Celcius]')

plt.show()



