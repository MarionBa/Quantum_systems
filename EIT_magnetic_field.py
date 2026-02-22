import numpy as np
import Functions as fct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


""" Magnetic field case - Section 4.2 of the reference paper 
H-J Su et al. "Optimizing the Rydberg EIT spectrum in a thermal vapor" Optics Express 30,2 (2022) 
This code has variations to what is described in the reference paper:
1- We enforce 0 Zeman splitting for transition 3/2 (small EIT peak) to fit ref. experimental results Fig. 3(a)
aka Delta_g=Delta_e=Delta_r=0 for Sigma_25 and Sigma_710
2- We use an ad-hoc Lande factors for the 5/3 transitions gF_r
3- We use an ad-hoc booster for the coupling Rabi frequency to fit the experimental/theoretical results, Eq.3, of the ref. Boost = 7.8"""

### Saving directory, for figures ###
dir = r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\Rydberg atom\Results EIT peak optimization'

### General parameters of the system ###
Boost = 7.8
# Parameters in ref. paper for Fig. 3
Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 0.36 # Optical density [1]
Gamma_e = 2 * np.pi * 60 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 5 # Spontaneous emission Rydberg state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
Omega_c_CGC1 = Boost * 2 * np.pi * 2.1 # The Rabi frequency of the coupling laser for CGC = 1 [MHz]
gamma = 2 * np.pi * 6 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe [MHz]
# Magnetic parameters as in ref. paper
gF_g = 1/2 # lande factor ground state
gF_e = 2/3 # lande factor excited state
mu_B_over_hbar = 1 # [MHz/G]

# External applied magnetic field
Mag_field = np.linspace(0, 25, 6) # [Gauss] external magnetic field

# Initialization arrays
T_array = []
Peak_split1 = []
Peak_split2 = []

# Loop over the magnetic field
n = 0
for B in Mag_field:
    n = n+1

    print(fr'Computing {(n-1)*100/(len(Mag_field)-1)}% done')
    print(fr'External magnetic field: {B} Gauss')


    ################################################################
    ################################################################

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
    sigma_23 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)

    ### |6> --> |7> --> |8> transition ###

    # Magnetic components
    mF_g = -2
    Delta_g = mu_B_over_hbar * gF_g * mF_g
    mF_e = -3
    Delta_e = mu_B_over_hbar * gF_e * mF_e
    mF_r = -4
    Delta_r = mu_B_over_hbar * gF_r * mF_r

    # Absorption cross-section
    sigma_78 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)

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
    sigma_24 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)


    ### The |6> --> |7> --> |9> transition ###

    # Magnetic components
    mF_g = -2
    Delta_g = mu_B_over_hbar * gF_g * mF_g
    mF_e = -3
    Delta_e = mu_B_over_hbar * gF_e * mF_e
    mF_r = -2
    Delta_r = mu_B_over_hbar * gF_r * mF_r

    # Absorption cross-section
    sigma_79 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)


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
    sigma_25 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)

    ### The |6> --> |7> --> |10> transition ###

    # Magnetic components
    Delta_g = 0
    Delta_e = 0
    Delta_r = 0

    # Absorption cross-section
    sigma_710 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)


    #######################################################################
    #######################################################################

    ### Total transmission - \sgima_eg = product of sigma_ij ###
    T = np.exp(-alpha*(np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)*np.array(sigma_78)*np.array(sigma_79)*np.array(sigma_710)))
    T_array.append(T)

    #######################################################################
    #######################################################################

    ### Extracting EIT peaks splitting ###

    peaks, _ = find_peaks(T, height=0) #(max(T)-min(T))/2)

    # Difference between peak splitting at zero magnetic field
    if B == 0:
        Peak_split1.append(0)
        Peak_split2.append(0)
    elif B == 5:
        Peak_split1.append(Delta_c[peaks[2]]-Delta_c[peaks[1]])
        Peak_split2.append(0)
    else:
        Peak_split1.append(Delta_c[peaks[4]]-Delta_c[peaks[1]])
        Peak_split2.append(Delta_c[peaks[3]]-Delta_c[peaks[2]])


    #######################################################################
    #######################################################################



### Figures ###


# Figure 1: spectra with the EIT peaks for different values of external magnetic field
plt.figure()
for i in range(len(Mag_field)):
    plt.plot(Delta_c, (T_array[i]-min(T_array[i]))*40 + i*int(max(Mag_field)/len(Mag_field)), color='tab:blue')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]', fontsize = 15)
plt.ylabel(r'Magnetic field [Gauss]', fontsize = 15)
plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
plt.title(fr'Transmission spectra', fontsize = 15)
plt.savefig(dir + r'\Magnetic field\Spectra.png')


# Figure 2: peak splitting
plt.figure(figsize=(8, 5))
# Linear fit of the peaks splitting points
a, b = np.polyfit(Mag_field, np.array(Peak_split1), 1)
plt.plot(Mag_field, a*Mag_field+b, color='k', linewidth=3)

a, b = np.polyfit(Mag_field[2:len(Peak_split2)], np.array(Peak_split2[2:len(Peak_split2)]), 1)
plt.plot(Mag_field, a*Mag_field+b, color='g', linewidth=3)

# Scatter plots
plt.scatter(Mag_field, np.array(Peak_split1), color='k')
plt.scatter(Mag_field[2:len(Peak_split2)], np.array(Peak_split2[2:len(Peak_split2)]), color='g')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Labels
plt.ylabel(r'Splitting [MHz]', fontsize = 15)
plt.xlabel(r'Magnetic field [Gauss]', fontsize = 15)
plt.legend([r'$|1> \to |2> \to |3>$', r'$|1> \to |2> \to |4>$'])
plt.title(fr'Peaks splitting with respect to magnetic field', fontsize = 15)
plt.savefig(dir + r'\Magnetic field\Peak_splitting.png')


plt.show()



